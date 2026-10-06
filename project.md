# PROJECT HANDBOOK
## PSO-Based Random Forest Hyperparameter Optimization — Complete Study Guide

**Anush Ramesh** · CB.EN.U4ELC23005 · EEE · Amrita Vishwa Vidyapeetham, Coimbatore
**Course:** Evolutionary Optimization (mini-project)
**Repository:** `C:\Users\anush\Desktop\PSO`
**Experiment of record:** `results/20260925-083324_default` (git commit `8bd5927`, clean tree)

> Every number in this handbook was read from a saved result file in that experiment directory.
> Nothing here is estimated, rounded up, or invented. File citations are given throughout.

---

# TABLE OF CONTENTS

| Part | Topic |
|---|---|
| **0** | [The 60-second answer](#part-0--the-60-second-answer) |
| **1** | [The problem and why it matters](#part-1--the-problem-and-why-it-matters) |
| **2** | [Theory: Random Forest](#part-2--theory-random-forest) |
| **3** | [Theory: Particle Swarm Optimization](#part-3--theory-particle-swarm-optimization) |
| **4** | [Theory: Cross-validation and data leakage](#part-4--theory-cross-validation-and-data-leakage) |
| **5** | [Mathematical formulation](#part-5--mathematical-formulation) |
| **6** | [The closed loop (the academic core)](#part-6--the-closed-loop-the-academic-core) |
| **7** | [Implementation walkthrough](#part-7--implementation-walkthrough) |
| **8** | [Experimental protocol](#part-8--experimental-protocol) |
| **9** | [Results — every number](#part-9--results--every-number) |
| **10** | [Answers to the research questions](#part-10--answers-to-the-research-questions) |
| **11** | [Verification, tests and proofs](#part-11--verification-tests-and-proofs) |
| **12** | [Slide-by-slide presentation script](#part-12--slide-by-slide-presentation-script) |
| **13** | [Viva questions and answers](#part-13--viva-questions-and-answers) |
| **14** | [Numbers to memorize](#part-14--numbers-to-memorize) |
| **15** | [Commands, file map, glossary](#part-15--commands-file-map-glossary) |

---

# PART 0 — THE 60-SECOND ANSWER

If someone stops you in the corridor and asks "what is your project?", say this:

> I built Particle Swarm Optimization from scratch in NumPy, and used it as a **controller** that tunes three
> Random Forest hyperparameters. It is a genuine **closed feedback loop**: PSO proposes a hyperparameter
> vector, a Random Forest is trained with it, its 5-fold cross-validation accuracy is measured, and that
> measurement is fed back as fitness so PSO updates its velocities and positions. Twenty iterations,
> ten particles, 210 evaluations per run. I ran it on three datasets — Iris, Digits and Heart Disease
> (UCI Cleveland) — with nested 5-fold cross-validation, and compared it against the default Random Forest
> and against an **equal-budget random search**, which is the open-loop control.
>
> The honest result: PSO reliably maximizes the objective it is given — its validation fitness is the highest
> of all three methods on all three datasets — but that does **not** translate into better held-out test
> accuracy. Against the default RF it is 4 wins, 6 ties, 5 losses over 15 folds. The Random Forest was already
> near its ceiling on these datasets, and the validation signal is noisy enough that the extra fitness is
> partly the winner's curse. That null result is the finding, and I can prove the loop and the test isolation
> with 203 passing tests.

That paragraph contains the whole project. Everything below is the detail behind it.

---

# PART 1 — THE PROBLEM AND WHY IT MATTERS

## 1.1 What is a hyperparameter?

A machine learning model has two kinds of numbers inside it:

| Kind | Who sets it | Example in Random Forest |
|---|---|---|
| **Parameter** | Learned from data during training | The split thresholds at each node of each tree |
| **Hyperparameter** | Set by the human *before* training | How many trees, how deep they can grow |

Parameters are fitted automatically. Hyperparameters are not — they are a **design choice**, and a bad choice
gives a worse model even with perfect data. There is no gradient to follow, because you cannot differentiate
"accuracy" with respect to "number of trees" — changing the number of trees from 100 to 101 either changes the
accuracy by a discrete jump or does not change it at all.

So hyperparameter tuning is a **black-box, derivative-free, expensive optimization problem**. That is exactly
the class of problem evolutionary and swarm methods exist for.

## 1.2 Why this is a hard search problem

Our search space has exactly **25,821 valid configurations** (derivation in §5.3). Evaluating one
configuration means training 5 Random Forests (one per inner CV fold) and averaging their accuracy. Measured
median cost per evaluation in this project (`results/20260925-083324_default/*/*/*/evaluations.csv`, reported
in `docs/DECISIONS.md` ADR-005 amendment):

| Dataset | Median seconds per evaluation |
|---|---|
| Iris | 0.354 s |
| Digits | 1.000 s |
| Heart (Cleveland) | 0.350 s |

Exhaustive grid search over all 25,821 points would therefore cost roughly **2.5 hours on Iris, 7.2 hours on
Digits and 2.5 hours on Heart** — and that is *per outer fold*. Across five outer folds and three datasets:

$$5 \times (2.5 + 7.2 + 2.5) \;\approx\; \mathbf{61\ hours}$$

Our PSO budget is 210 evaluations, which is $210 / 25{,}821 = \mathbf{0.81\ \%}$ of the space, and the whole
experiment — all three datasets, all five folds, all three methods — finished in **62.7 minutes**.

That ratio — 0.81 % of the space, 62.7 minutes instead of ~61 hours, roughly a **58× speed-up** — is the
practical motivation for the project.

## 1.3 Why Random Forest as the system being optimized

From `docs/DECISIONS.md` ADR-002:

- It is the system named in the original project proposal (PDF slide 6).
- Its hyperparameters are **few and interpretable** — you can explain what each one does in one sentence, which
  matters for a presentation.
- It is **fast** on these datasets, so a 210-evaluation loop is feasible on a laptop.
- It needs **no feature scaling**, which removes a whole class of preprocessing leakage worries.
- It is **stochastic but seedable**, so with a fixed `random_state` the fitness function is deterministic — and
  determinism is what makes the personal-best comparison fair and the cache exact.

## 1.4 Why PSO as the optimizer

From `docs/DECISIONS.md` ADR-001, with the alternatives that were considered and rejected:

| Candidate | Verdict |
|---|---|
| **PSO** | **Chosen.** Derivative-free, handles a bounded continuous box natively, tiny implementation, shares information between particles so a small swarm is efficient, and has an explicit exploration/exploitation balance that is easy to explain. |
| Genetic Algorithm | Workable, but needs crossover/mutation operators designed for mixed integer ranges, and has more knobs to justify. |
| Ant Colony Optimization | Designed for path and combinatorial problems; a box of three bounded integers is not its natural domain. |
| Grid search | 25,821 configurations — impractical (see §1.2). |
| Random search | **No feedback.** It is used here as the *open-loop control*, not as the optimizer. This is the scientific contrast of the project. |
| Bayesian optimization | Genuinely strong for expensive black boxes, but outside the evolutionary scope of the course. |

The key sentence for the viva: **PSO was chosen because the fitness has no gradient, each evaluation is
expensive, and the space is a low-dimensional bounded box — which is precisely PSO's design envelope.**

## 1.5 The academic requirement this project satisfies

The faculty brief requires:

1. One optimization technique maximizing a defined objective → **PSO maximizing mean inner-CV accuracy**.
2. The optimizer's output must change a **system parameter** → **the three RF hyperparameters**.
3. There must be **closed-loop feedback**, with the optimizer as its own block → **Part 6**.
4. Review 2 must show the whole system as a closed loop with the output fed back → the per-iteration console
   trace and the live frontends.

---

# PART 2 — THEORY: RANDOM FOREST

You must be able to explain the *plant* you are controlling. This part is the theory of that plant.

## 2.1 From one decision tree to a forest

A **decision tree** splits the feature space by asking yes/no questions ("is petal length < 2.45 cm?"), choosing
at each node the split that most reduces impurity. Grown without limit, a tree will keep splitting until every
leaf is pure — it memorizes the training set. That is **low bias, very high variance**: change a few training
rows and the tree changes shape completely.

A **Random Forest** fixes the variance problem with two independent sources of randomness:

### Source 1 — Bagging (bootstrap aggregating)

Each tree is trained on a **bootstrap sample**: *n* rows drawn *with replacement* from the *n* training rows.
Each bootstrap sample therefore contains roughly 63.2 % of the distinct original rows, and duplicates of some.

Why 63.2 %? The probability that a specific row is *never* picked in *n* independent draws is

$$\left(1 - \frac{1}{n}\right)^{n} \xrightarrow{n \to \infty} e^{-1} \approx 0.368$$

so the expected fraction *included* is $1 - e^{-1} \approx 0.632$. The ~36.8 % left out of a given tree are that
tree's **out-of-bag** rows.

### Source 2 — Random feature subsampling

At **every split of every tree**, only a random subset of features is considered as candidates. In
scikit-learn's classifier the default is $\sqrt{d}$ features, where *d* is the total number of features.

This second source is what makes a Random Forest more than just bagged trees. Without it, if one feature is
strongly predictive, every tree would split on it first, and the trees would be highly correlated — averaging
correlated predictors barely reduces variance.

### Why averaging works — the variance argument

If you average *B* predictors each with variance $\sigma^2$ and pairwise correlation $\rho$, the variance of the
average is

$$\mathrm{Var}\!\left(\bar{h}\right) = \rho\,\sigma^{2} + \frac{1 - \rho}{B}\,\sigma^{2}$$

Read this equation carefully, because it explains **both** hyperparameters that matter most:

- The second term $\frac{1-\rho}{B}\sigma^2$ **vanishes as $B \to \infty$**. More trees always helps, with
  diminishing returns. This is why accuracy is roughly **non-decreasing in `n_estimators`** — and why
  `n_estimators` behaves more like a *cost knob* than a *tuning knob*. We predicted and recorded this honestly
  in ADR-003.
- The first term $\rho\sigma^2$ **does not vanish**. It is the floor. Reducing correlation $\rho$ (feature
  subsampling) is the only way to lower it. Interesting consequence: `max_features`, which directly controls
  $\rho$, is often RF's single most influential hyperparameter — and **we do not tune it**, because the project
  proposal fixed the three decision variables. We state this limitation openly (ADR-003, critique point 3).

**Prediction rule:** each tree votes for a class; the forest returns the **majority vote** (scikit-learn
actually averages the per-tree predicted probabilities, which is a soft version of the same idea).

## 2.2 The three hyperparameters we tune

| Hyperparameter | Range | What it physically controls | Effect of increasing it |
|---|---|---|---|
| **`n_estimators`** | 50 – 200 | Number of trees *B* in the ensemble | Lower variance, diminishing returns, **linearly more compute** |
| **`max_depth`** | 2 – 20 | Maximum levels any tree may grow | More expressive (lower bias), more overfitting risk (higher variance) |
| **`min_samples_split`** | 2 – 10 | Minimum samples a node must have to be split further | **Regularizes** — larger value = shallower, smoother trees, less overfitting |

Note the direction of the third one carefully, because it is easy to get backwards in a viva:
**larger `min_samples_split` means *less* complex trees.** A node with 9 samples cannot be split if
`min_samples_split = 10`, so growth stops earlier.

`max_depth` and `min_samples_split` are both complexity controls, but they act differently: `max_depth` is a
*global* cap on levels, while `min_samples_split` is a *local* cap based on how much data reaches a node. They
interact — which is one reason the fitness landscape is not separable and a joint search is justified.

## 2.3 The baseline, and why it sits outside the search space

The comparison baseline is `RandomForestClassifier()` with scikit-learn's defaults
(`src/pso_rf/models/baseline.py`):

$$\theta_0 = (n\_estimators = 100,\; max\_depth = \texttt{None},\; min\_samples\_split = 2)$$

`max_depth = None` means **unlimited depth** — trees grow until leaves are pure. Since our search space caps
depth at 20, **the baseline is not a point PSO can reach**. This is deliberate (ADR-008) and you must be ready
to defend it:

- It is the honest real-world comparison: it is what a practitioner gets by typing `RandomForestClassifier()`.
- It means a PSO "win" is a real win against standard practice, not against a hobbled competitor.
- It also means a PSO "loss" may be caused by the **bound**, not by the optimizer. On Digits the default RF
  grows unlimited-depth trees, and our cap of 20 may be binding. We flagged this risk in ADR-003 *before*
  running, and the Digits results (§9.4) are consistent with it.

## 2.4 Why the fitness surface has no gradient

Accuracy is a count of correct predictions divided by the number of samples. It is therefore a **step
function**: on Iris, where each outer fold's optimization portion has 120 rows and each sample is validated
exactly once across the 5 inner folds, the fitness can only move in multiples of roughly $1/120 \approx
0.0083$.

Consequences, all of which show up in our results:

1. **No derivative exists** → gradient descent is impossible → derivative-free search is mandatory.
2. **Massive ties.** Many different configurations give *exactly* the same fitness. This is why tie-breaking
   had to be specified precisely (ADR-017), and why `n_ties_with_best` is a recorded field — on Heart fold 0,
   **6 distinct configurations tied** with the winner (`heart_cleveland/fold_0/pso/final.json`).
3. **Plateaus.** The gbest curve is flat for long stretches. On Iris, PSO's best configuration was often found
   at **iteration 0** — i.e. in the initial random swarm — because the surface is nearly flat at the top.

---

# PART 3 — THEORY: PARTICLE SWARM OPTIMIZATION

This is the heart of the project. Know this part cold.

## 3.1 The origin and the metaphor

PSO was introduced by **Kennedy and Eberhart in 1995**. It came out of simulations of **bird flocking and fish
schooling**. The observation: a flock finds food efficiently not because any bird knows where the food is, but
because each bird balances two instincts — *go back to where I personally found the most food* and *go towards
where the flock collectively found the most food*.

Translate that into optimization:

- A **particle** is a candidate solution — a point in the search space. Here, a hyperparameter vector.
- Each particle has a **position** (its current candidate) and a **velocity** (the step it is about to take).
- Each particle remembers its own best-ever position: **pbest** (the *cognitive* memory).
- The swarm shares the best position found by anyone: **gbest** (the *social* memory).
- Every iteration, each particle's velocity is pulled towards both memories, with random weights.

The crucial design property: **a particle's next move depends on measured fitness values.** That is the feedback
that makes this a closed loop rather than a sampling scheme.

## 3.2 The three forces — the intuition you must be able to speak

Every velocity update is the sum of exactly three terms:

```
            ┌─────────── INERTIA ───────────┐
            │  w · v                        │   "keep going the way I was going"
            │                               │   → exploration, momentum, escaping local optima
            └───────────────────────────────┘
            ┌────────── COGNITIVE ──────────┐
            │  c₁ · r₁ · (p − x)            │   "pull me back to MY best"
            │                               │   → individual memory, independent search
            └───────────────────────────────┘
            ┌─────────── SOCIAL ────────────┐
            │  c₂ · r₂ · (g − x)            │   "pull me towards the SWARM's best"
            │                               │   → information sharing, exploitation, convergence
            └───────────────────────────────┘
```

A one-line analogy that works well out loud: *inertia is habit, the cognitive term is personal experience, and
the social term is peer pressure.*

Each force is a **spring-like attraction**: the magnitude $(p - x)$ grows with distance, so a particle far from
the best is pulled harder. As the swarm converges, $(p-x)$ and $(g-x)$ both shrink, so the steps shrink
automatically — PSO self-anneals without an explicit cooling schedule.

## 3.3 The equations

For particle *i*, dimension *d*, at iteration *t*:

**Velocity update**

$$\tilde{v}_{i,d}^{\,t+1} = \underbrace{w\,v_{i,d}^{\,t}}_{\text{inertia}} + \underbrace{c_1 r_{1,i,d}\left(p_{i,d}^{\,t} - x_{i,d}^{\,t}\right)}_{\text{cognitive}} + \underbrace{c_2 r_{2,i,d}\left(g_{d}^{\,t} - x_{i,d}^{\,t}\right)}_{\text{social}}$$

**Velocity clamp**

$$v_{i,d}^{\,t+1} = \mathrm{clip}\!\left(\tilde{v}_{i,d}^{\,t+1},\; -v_{\max,d},\; +v_{\max,d}\right)$$

**Position update**

$$x_{i,d}^{\,t+1} = x_{i,d}^{\,t} + v_{i,d}^{\,t+1}$$

Three details that are easy to lose marks on:

- $r_1$ and $r_2$ are drawn **fresh for every particle, every dimension, every iteration** — they are *vectors*,
  not scalars. Using one scalar per iteration restricts movement to a line and is a classic implementation bug.
  Our code draws `R1 = rng.random((N, D))` and `R2 = rng.random((N, D))` — full matrices
  (`src/pso_rf/optimization/pso.py:165-166`).
- The pulls use **element-wise multiplication**, so each dimension gets an independent random weight.
- Velocity is **not** a physical speed; it is just the increment added to position.

## 3.4 The parameter values, and why these exact numbers

| Parameter | Value | Justification |
|---|---|---|
| $w$ (inertia) | **0.7298** | Clerc & Kennedy (2002) **constriction coefficient** |
| $c_1$ (cognitive) | **1.49618** | Same constriction set |
| $c_2$ (social) | **1.49618** | Same constriction set — equal weight to self and swarm |
| $N$ (swarm size) | **10** | From the project proposal (PDF slide 10); ADR-014 |
| $T$ (iterations) | **20** | From the proposal; gives $N(T+1) = 210$ evaluations |
| $v_{\max}$ | **0.2 × range** = (30, 3.6, 1.6) | ADR-012 |
| $v_{\text{init}}$ | **0.1 × range** = (15, 1.8, 0.8) | ADR-012 |

### Where 0.7298 and 1.49618 come from (be ready for this question)

These are **not** arbitrary. They are the **Clerc–Kennedy constricted PSO** values, derived from a stability
analysis of the PSO difference equation. The construction is:

$$\chi = \frac{2}{\left|2 - \varphi - \sqrt{\varphi^{2} - 4\varphi}\right|}, \qquad \varphi = \varphi_1 + \varphi_2 = 4.1$$

Evaluating with $\varphi = 4.1$ gives $\chi \approx 0.72984$, and then

$$w = \chi \approx 0.7298, \qquad c_1 = c_2 = \chi \cdot \frac{\varphi}{2} = 0.72984 \times 2.05 \approx 1.49618$$

**Why it matters:** the constriction factor guarantees the swarm's trajectory is **asymptotically stable** — it
converges instead of oscillating with growing amplitude. With $w$ too large (say 1.2) the swarm explodes; with
$w$ too small (say 0.3) it collapses onto gbest almost immediately and stops exploring. $\varphi = 4.1$ (just
above the critical value 4) is the standard recommendation, and 0.7298/1.49618 is the single most widely used
PSO parameter set in the literature.

Short answer for the viva: *"They are the Clerc–Kennedy constriction coefficients with φ = 4.1, which make the
swarm provably stable — it converges rather than diverging. They are the literature-standard canonical PSO
values, so I did not have to tune the tuner."*

That last clause is itself worth saying: **we did not tune PSO's own hyperparameters.** If we had, we would be
tuning a tuner against validation data, which would make the comparison against random search unfair.

### Why clamp the velocity

Without $v_{\max}$, a particle far from gbest gets a huge pull and can jump across the entire box in one step,
turning the search into random sampling. Clamping at 0.2 × range means **no particle can cross more than 20 %
of a dimension's span in a single iteration**, which preserves the local, trajectory-like character of the
search. The three clamp values follow from the ranges: $200-50 = 150$, $20-2 = 18$, $10-2 = 8$, so
$v_{\max} = 0.2 \times (150, 18, 8) = (30,\, 3.6,\, 1.6)$.

## 3.5 Boundary handling — the absorbing wall

Particles will try to leave the box. Our rule (ADR-012, implemented in
`src/pso_rf/optimization/pso.py:79-84`):

$$\left(x_{i,d}^{t+1},\, v_{i,d}^{t+1}\right) = \begin{cases} (l_d,\; 0) & \text{if } \tilde{x}_{i,d}^{t+1} < l_d \\ (u_d,\; 0) & \text{if } \tilde{x}_{i,d}^{t+1} > u_d \\ \left(\tilde{x}_{i,d}^{t+1},\; v_{i,d}^{t+1}\right) & \text{otherwise} \end{cases}$$

In words: **clamp the position to the wall, and set that dimension's velocity to zero.**

```python
def apply_absorbing_bounds(X_new, V, lower, upper):
    outside = (X_new < lower) | (X_new > upper)
    return np.clip(X_new, lower, upper), np.where(outside, 0.0, V)
```

Why zero the velocity? If you only clamped the position, the particle would keep its large outward velocity and
spend every subsequent iteration pressed against the wall, wasting evaluations. Zeroing lets the cognitive and
social terms turn it around on the next iteration.

Why *absorb* rather than *reflect* (bounce it back with reversed velocity)? Because **the optimum may legitimately
be on the boundary**. Accuracy is roughly non-decreasing in `n_estimators`, so the true best is plausibly at
`n_estimators = 200`. An absorbing wall lets a particle **sit on** the bound; a reflecting wall would push it
away from the very point we want. This prediction was confirmed: on Heart fold 0, PSO selected
`n_estimators = 200` exactly, and `final.json` records `"boundary_hits": ["n_estimators"]`.

Reflecting boundaries are left as a documented, unimplemented extension — `PSOOptimizer.__init__` raises
`NotImplementedError` for anything other than `absorb`/`gbest`/`synchronous`, so the code cannot silently run a
variant we never validated.

## 3.6 Topology: gbest (fully connected)

The **topology** decides which neighbours a particle learns from.

| Topology | Structure | Behaviour |
|---|---|---|
| **gbest** (ours) | Every particle sees the single global best | Fast information spread, fast convergence, higher risk of premature convergence |
| lbest / ring | Each particle sees only *k* neighbours | Slower spread, better diversity, better on multimodal problems |

We use **gbest** (ADR-014) because with only $N = 10$ particles and $T = 20$ iterations, the budget is far too
small to afford slow information propagation. In a ring topology, news of a good configuration would take
several iterations to cross the swarm — and we only have 20.

## 3.7 Synchronous vs asynchronous update

- **Synchronous (ours):** evaluate *all* particles, *then* update all pbests, *then* update gbest, *then* move.
- **Asynchronous:** update gbest immediately after each particle is evaluated, so later particles in the same
  iteration already use it.

We chose synchronous because it is **deterministic and order-independent**, which makes the run exactly
reproducible and the parallel evaluation of particles safe. It is also the form in the textbook pseudocode, so it
is easier to defend.

## 3.8 Tie-breaking and the strict inequality (ADR-017)

Because the fitness surface is full of exact ties (§2.4), the comparison operator matters:

- pbest is replaced **only** on strict improvement: `F > P_fit`, never `>=`.
- gbest is replaced **only** on strict improvement.
- On equal fitness, the **lowest particle index** wins (`np.argmax` returns the first maximum).

Why strict `>` and not `>=`? Three reasons:

1. **Stability.** With `>=`, a swarm sitting on a plateau would keep swapping its best position between
   equal-fitness points, so gbest would jitter and the convergence curve would be meaningless.
2. **Monotonicity.** With strict `>`, $\gamma^t$ (the gbest fitness) is **provably non-decreasing** in *t*. The
   convergence curve can never go down. That is what makes it a valid convergence plot.
3. **Reproducibility.** Deterministic tie-breaking means the same seed gives the same trajectory, every time.

## 3.9 The integer problem and the decode operator (ADR-011)

Our hyperparameters are integers, but PSO works in continuous space. The solution:

$$\mathrm{dec}(x) = \mathrm{clip}\!\left(\mathrm{rint}(x),\, l,\, u\right)$$

**The swarm state $(x, v, p, g)$ stays continuous at all times. Rounding happens only at the moment of
evaluation, and the rounded value is never written back into the state.**

This is one of the most important design decisions in the project, and a very likely viva question.

**Why not just round the state itself?** Suppose a particle sits at `max_depth = 8.0` and its velocity in that
dimension is 0.3. If we rounded the state, the particle would move to 8.3, round back to 8, be stored as 8, and
next iteration start from 8 again — **it would be frozen forever** in that dimension, because every step smaller
than 0.5 is annihilated. Keeping the state continuous lets small velocity contributions **accumulate**: 8.0 →
8.3 → 8.6 → 8.9 → 9.2, and at 8.6 the decoded value finally flips from 8 to 9. The particle genuinely explores.

A second detail worth knowing: `np.rint` uses **banker's rounding** (ties go to the *even* integer), so
$\mathrm{rint}(8.5) = 8$ and $\mathrm{rint}(9.5) = 10$. This is unbiased over many roundings, unlike always
rounding half up.

A consequence of decoding is that **distinct positions can decode to the same configuration**, which is why the
fitness cache is effective and why `n_unique_fits` is less than 210 for PSO (§9.6) — that gap is itself evidence
of convergence.

## 3.10 Stopping and the definition of convergence (ADR-015)

- **Default stopping:** run the full $T = 20$ iterations. `stop_reason = "max_iter"`.
- **Optional patience rule (disabled by default):** stop at the first $t \ge P$ where
  $\gamma^{t} - \gamma^{t-P} < \mathrm{tol}$, with $P = 5$, $\mathrm{tol} = 10^{-4}$.

Patience is **off** in the experiment so that every run has exactly 210 evaluations and the comparison with
random search is budget-exact. The feature exists, is tested, and is reported as available.

**Convergence iteration** is recorded for every run:

$$t_c = \max\{\,t : \gamma^{t} > \gamma^{t-1}\,\}$$

the **last** iteration at which gbest improved (0 if it never improved after initialization). Each PSO run is
also labelled (`src/pso_rf/experiments/runner.py:322-326`):

- `improving_at_end` — gbest improved within the last 3 iterations (the budget may have been too small)
- `plateaued` — it did not (the search had settled)

## 3.11 Full algorithm, as implemented

```
input: space (l, u), objective f, N=10, T=20, w, c1, c2, v_max, rng

X ← U(l, u)                              (N×3)  initial positions
V ← U(−0.1(u−l), +0.1(u−l))              (N×3)  initial velocities
F ← [f(dec(X_i)) for all i]              ← FEEDBACK from the plant (t = 0)
P ← X ;  Pf ← F                                 personal bests
g ← P[argmax Pf] ;  gf ← max Pf                 global best

for t = 1 … T:
    R1, R2 ← rng.random((N,3)), rng.random((N,3))        fresh randomness
    V ← clip(w·V + c1·R1·(P − X) + c2·R2·(g − X), ±v_max)   velocity update
    X̃ ← X + V                                            position update
    out ← (X̃ < l) | (X̃ > u)
    X ← clip(X̃, l, u) ;  V[out] ← 0                      absorbing wall
    F ← [f(dec(X_i)) for all i]           ← NEW PARAMETERS SENT, NEW FEEDBACK RETURNED
    improved ← F > Pf                                     strict improvement only
    P[improved] ← X[improved] ;  Pf[improved] ← F[improved]
    i* ← argmax Pf
    if Pf[i*] > gf:  g ← P[i*] ;  gf ← Pf[i*]            strict improvement only
    record iteration summary ;  check optional patience

return dec(g), gf
```

The two lines marked with arrows are the closed loop. Everything else is bookkeeping.

## 3.12 Exploration vs exploitation, and how we measured it

- **Exploration** = searching new regions. Driven by inertia $w$ and by the swarm being spread out.
- **Exploitation** = refining what is already good. Driven by $c_1, c_2$ and by the swarm being clustered.

PSO manages the transition **automatically**: early on, particles are far apart, so $(p-x)$ and $(g-x)$ are large
and the steps are big. As the swarm converges, those distances shrink and the steps shrink with them.

We **measure** this directly with a diversity metric (`src/pso_rf/optimization/pso.py:103-106`): the mean
Euclidean distance of the range-normalized positions from their centroid.

```python
def swarm_diversity(X, lower, range_):
    Z = (X - lower) / range_
    return float(np.mean(np.linalg.norm(Z - Z.mean(axis=0), axis=1)))
```

Normalizing by the range first is essential — otherwise `n_estimators` (range 150) would dominate
`min_samples_split` (range 8) and the number would be meaningless. Diversity is logged every iteration into
`iterations.csv` and plotted as **figure F11**. A falling diversity curve is the visual proof that the swarm is
contracting, i.e. exploration giving way to exploitation.

---

# PART 4 — THEORY: CROSS-VALIDATION AND DATA LEAKAGE

This part is where projects lose the most marks, and where this project is strongest. The entire architecture is
built around it.

## 4.1 Why a single train/test split is not enough

If you tune hyperparameters by maximizing accuracy on a test set, and then *report* that test accuracy, your
number is **meaningless**. You selected the configuration *because* it scored well on that data, so the score is
biased upward. You have leaked the test set into the model-selection process.

The fix is **three** roles for data, not two:

| Role | Purpose | Who may see it |
|---|---|---|
| **Training** | Fit the forest's parameters | The model |
| **Validation** | Compute the fitness that PSO optimizes | PSO (via fitness only) |
| **Test** | Produce the one honest reported number | Nobody, until the loop has terminated |

## 4.2 k-fold cross-validation

Split the data into *k* equal stratified parts. For each part *j*: train on the other $k-1$ parts, validate on
part *j*. Average the *k* scores.

- **Stratified** means each fold preserves the class proportions of the whole set. Essential for small or
  imbalanced data — on Iris, a non-stratified fold could by chance contain almost no *virginica*.
- Every sample is validated **exactly once**, so the averaged accuracy has resolution $1/n$ for any *k*.

## 4.3 Nested cross-validation — our structure

```
                       FULL DATASET  (e.g. Iris, 150 rows)
                               │
          StratifiedKFold(5, shuffle=True, random_state=42)
                               │
   ┌───────────┬───────────┬───┴───────┬───────────┬───────────┐
 fold 0      fold 1      fold 2      fold 3      fold 4      ← OUTER
   │
   │  For run k = 0:
   ├── TEST  = fold 0        (30 rows)  ── SEALED, untouched until the loop ends
   └── OPT   = folds 1-4    (120 rows)
                 │
                 │  StratifiedKFold(5, shuffle=True, random_state=k)
                 │  ** drawn ONCE and FIXED for the whole run **
                 │
        ┌────┬────┬────┬────┬────┐
       Q1   Q2   Q3   Q4   Q5         ← INNER  (24 rows each on Iris)
        └──────── these are the VALIDATION data; their mean accuracy = fitness ────────┘
```

This is repeated for $k = 0,1,2,3,4$, so **every row of the dataset serves as test data exactly once**. That is
what makes the pooled confusion matrix (figure F10) legitimate.

### Why the inner folds are drawn once and frozen (ADR-005)

This is subtle and a likely question. If we re-drew the inner folds for every evaluation:

1. The fitness would become **stochastic** — the same configuration would return different values on different
   calls.
2. The **cache would be invalid**, since a cache hit would return a value computed on different data.
3. **pbest would reward luck.** A mediocre configuration that happened to get an easy fold split would be
   recorded as a personal best and would then pull its particle — and through gbest, the whole swarm — in the
   wrong direction.

Freezing the folds is the **common random numbers** technique from simulation optimization: every candidate is
judged on *identical* data, so differences between candidates reflect the candidates, not the sampling.

### Why 5 inner folds rather than the 3 in the original proposal

The original PDF said 3-fold. We changed it to 5-fold and recorded the change as ADR-005. The reasoning:

- Fitness resolution is $1/n$ regardless of *k*, so *k* does not change precision. What it changes is the
  **training-fold size**: 67 % of the optimization data for 3-fold, **80 %** for 5-fold.
- `max_depth` and `min_samples_split` act on **absolute sample counts**. Tuning them at 80 % of the data
  transfers better to the final refit, which uses 100 %.
- Lower variance in the fitness estimate.
- The measured extra cost is only **4–20 %** (ADR-005 timing gate), because the 5 folds run in parallel.

A **timing gate** was built into the plan: if the projected total runtime exceeded 2 hours, fall back to 3-fold
and record it. Measured projection was **37.5 minutes**, so 5-fold was kept. (The actual run took 62.7 min —
longer than projected but still well under the 2-hour gate, and the discrepancy is documented honestly in the
ADR-005 amendment rather than hidden.)

## 4.4 How leakage is made structurally impossible

This project does not rely on being careful. It makes leakage **raise an exception**. Four independent
mechanisms (ADR-024):

**1. The test arrays are private.** `HeldOutTestSet` stores them in name-mangled attributes
(`self.__X`), reachable only through `reveal()` (`src/pso_rf/evaluation/splits.py:92-123`).

**2. `reveal()` refuses to work during optimization.**

```python
def reveal(self) -> tuple[np.ndarray, np.ndarray]:
    if optimization_phase_active():
        raise TestSetAccessError(
            f"held-out test data (n={len(self)}) requested while an OptimizationPhase is open; "
            "test data may be read only by the final evaluation, after optimization ends"
        )
    log.debug("held-out test set revealed (n=%d)", len(self))
    return self.__X.copy(), self.__y.copy()
```

**3. Every optimization is wrapped in that phase.** In `runner.py:140-142`:

```python
evaluator = _make_evaluator(cfg, settings, fold.opt, seeds, metric, log)
with OptimizationPhase():                       # test fold is now sealed
    outcome = _optimize(method, cfg, settings, evaluator, context, out_dir, seeds, log, callbacks)
optimization_finished_at = utc_timestamp()      # phase closed — only now may test data be read
final = final_evaluate(outcome.config, fold.opt, fold.test, ...)
```

**4. The optimizer never receives data at all.** `PSOOptimizer.__init__` takes a `SearchSpace`, an `objective`
callable, a `PSOConfig` and an RNG. There is no `X`, no `y`, no dataset name. It literally cannot leak what it
cannot see. The `FitnessEvaluator` constructor additionally **rejects** a `HeldOutTestSet` if one is ever passed.

The phase flag is **thread-local**, not process-wide. This was an amendment (ADR-024, 2026-09-25): the
interactive demo serves multiple browser sessions from one process on separate threads, and a process-wide flag
meant one session's live search blocked another session's legitimate final evaluation. A phase guards *the flow
of control that opened it*. The parallel CV folds run in separate processes that never touch test data.

### The temporal proof in the log

Because `run.log` timestamps every phase transition, you can **prove** the ordering after the fact. From
`results/20260925-083324_default/run.log`:

```
08:33:24.577  DEBUG  optimization phase opened (depth 1)
08:33:26.094  INFO   [iris|fold 0|baseline|seed 0] baseline {...}: validation accuracy 0.9417
08:33:26.094  DEBUG  optimization phase closed (depth 0)
08:33:26.190  DEBUG  held-out test set revealed (n=30)
08:33:26.205  INFO   [iris|fold 0|baseline|seed 0] test accuracy 1.0000 (evaluated once, after optimization)
```

The reveal line is **always** after the close line. `verify_results` parses the whole log and checks this
automatically — one of the checks behind the `OK: all checks passed` verdict.

## 4.5 The winner's curse — why best validation fitness is never reported as performance

We evaluate 210 configurations and report the **maximum** validation fitness. Even if every configuration had
the *same* true accuracy, the maximum of 210 noisy estimates would be above the true value, purely by chance.
This is the **winner's curse** (or *optimization bias*).

So (ADR-020):

- Only **held-out test metrics** are ever called "performance".
- Best validation fitness is always labelled **"optimistically biased (selection over 210 evaluations)"**.

Our own data shows the gap. On Heart Disease (`summary.csv`):

| Quantity | Value |
|---|---|
| PSO mean validation fitness | **0.8415** |
| PSO mean held-out test accuracy | **0.8283** |
| Optimism gap | **+1.3 percentage points** |

If we had reported 0.8415 as "our accuracy", we would have overstated the result by 1.3 points. We report
0.8283.

## 4.6 Preprocessing leakage, and why the imputer is a pipeline step

The Heart Cleveland data has missing values: 4 in `ca` and 2 in `thal` (`heart_cleveland/audit.json`). We impute
with the **most frequent** value.

The naive mistake is to impute on the whole dataset and *then* split. That leaks: the mode is computed using
rows that later become validation or test data.

Our fix (ADR-010): the imputer is the **first step of a scikit-learn `Pipeline`**, so
`cross_validate` refits it **inside each fold, on that fold's training rows only**.

```python
# src/pso_rf/preprocessing/pipeline.py
def build_steps(spec):
    return [("impute", SimpleImputer(strategy=spec.impute))] if spec.impute else []
```

Unit test **UT-20** asserts the imputer's learned statistics come only from training rows.

## 4.7 The class-ratio gate

Accuracy is a bad metric for imbalanced data — on a 99:1 split, predicting the majority class always gives 99 %.
So the config has a gate (`configs/default.yaml`): if a dataset's max/min class ratio exceeds **1.5**,
automatically switch the fitness metric to **balanced accuracy** (the mean per-class recall).

Measured ratios (`*/audit.json`):

| Dataset | Class ratio | Gate (1.5) | Metric used |
|---|---|---|---|
| Iris | 1.000 | not triggered | accuracy |
| Digits | 1.052 | not triggered | accuracy |
| Heart Cleveland | 1.180 | not triggered | accuracy |

**No dataset triggered the gate**, so all three ran on plain accuracy. Balanced accuracy and macro F1 are still
computed and logged every evaluation as **diagnostics** — and are **never fed back to PSO**, because changing
the objective mid-run would invalidate the comparison.

---

# PART 5 — MATHEMATICAL FORMULATION

## 5.1 Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{D} = \{(\mathbf{a}_i, y_i)\}_{i=1}^{n}$ | the dataset |
| $P_k$ | outer fold *k* |
| $\mathcal{D}^{(k)}_{test} = P_k$ | held-out test data of run *k* |
| $\mathcal{D}^{(k)}_{opt} = \mathcal{D} \setminus P_k$ | optimization portion of run *k* |
| $Q^{(k)}_j$ | inner fold *j* of run *k* (the validation data) |
| $\theta$ | a hyperparameter configuration |
| $s_k = k$ | run seed of fold *k* |
| $x, v, p, g$ | position, velocity, personal best, global best |
| $\gamma^t$ | gbest fitness at iteration *t* |

## 5.2 Decision variables

$$\theta = (\theta_1, \theta_2, \theta_3) = (n\_estimators,\; max\_depth,\; min\_samples\_split)$$

## 5.3 Search space and constraints

$$l = (50,\, 2,\, 2), \qquad u = (200,\, 20,\, 10)$$

| Constraint | Meaning |
|---|---|
| $50 \le \theta_1 \le 200$ | number of trees |
| $2 \le \theta_2 \le 20$ | maximum tree depth |
| $2 \le \theta_3 \le 10$ | min samples to split; scikit-learn requires $\ge 2$, so this is also a **validity** constraint |
| $\theta \in \mathbb{Z}^3$ | integer-valued |

Feasible set and its size:

$$\Omega = \mathbb{Z}^3 \cap [l, u] = \{50..200\} \times \{2..20\} \times \{2..10\}$$

$$|\Omega| = 151 \times 19 \times 9 = \mathbf{25{,}821}$$

(Count inclusively: $200-50+1 = 151$, $20-2+1 = 19$, $10-2+1 = 9$.)

PSO searches the **continuous relaxation** $B = [l, u] \subset \mathbb{R}^3$.

## 5.4 The decode operator

$$\mathrm{dec}: B \to \Omega, \qquad \mathrm{dec}(x) = \mathrm{clip}\!\left(\mathrm{rint}(x),\, l,\, u\right)$$

## 5.5 The objective function

$$\boxed{\;F_k(\theta) \;=\; \frac{1}{5}\sum_{j=1}^{5} \mathrm{acc}\!\left(\mathrm{RF}_{\theta,\, s_k}\!\left(\mathcal{D}^{(k)}_{opt} \setminus Q^{(k)}_j\right);\; Q^{(k)}_j\right)\;}$$

where $\mathrm{acc}(h; S') = \frac{1}{|S'|}\sum_{(\mathbf{a},y) \in S'} \mathbb{1}[h(\mathbf{a}) = y]$.

The fitness of a continuous position is $f_k(x) = F_k(\mathrm{dec}(x))$.

**The optimization problem, for each outer fold *k*:**

$$\theta^{*}_k \in \arg\max_{\theta \in \Omega} F_k(\theta)$$

PSO returns the approximation $\hat{\theta}_k = \mathrm{dec}\!\left(g_k^{\text{final}}\right)$.

### Four properties of $F_k$ that drive the whole design

1. **Test independence.** $F_k$ is a function of $\mathcal{D}^{(k)}_{opt}$ *only*. Changing any value in
   $\mathcal{D}^{(k)}_{test}$ leaves $F_k$ — and therefore the entire PSO trajectory — bit-for-bit unchanged.
   Tested directly by IT-06 (labels) and IT-07 (features).
2. **Deterministic within a run.** Fixed inner folds plus fixed seed $s_k$ ⟹ the same $\theta$ always returns
   the same value. This makes caching exact and pbest comparisons fair.
3. **Piecewise constant.** Resolution $1/|\mathcal{D}^{(k)}_{opt}|$ — on Iris, $1/120$. Hence ties and plateaus,
   hence no gradient, hence PSO.
4. **Optimistically biased at the maximum.** $\max_\theta F_k(\theta)$ over 210 evaluations overestimates the
   true generalization accuracy. Never reported as performance (§4.5).

## 5.6 Final evaluation — strictly after the loop

Once optimization of fold *k* has **terminated** and $\hat{\theta}_k$ is **fixed**:

$$\hat{h}_k = \mathrm{RF}_{\hat{\theta}_k,\, s_k}\!\left(\mathcal{D}^{(k)}_{opt}\right) \qquad \text{(refit on the \emph{whole} optimization portion)}$$

$$M_k = m\!\left(\hat{h}_k;\; \mathcal{D}^{(k)}_{test}\right)$$

where *m* ranges over accuracy, balanced accuracy, macro precision, macro recall, macro F1, the confusion
matrix, and — for binary Heart — positive-class precision/recall/F1.

- $\mathcal{D}^{(k)}_{test}$ appears **only** here. Never in $F_k$, $\arg\max$, $p$, $g$, $v$, $x$, or the
  stopping rule.
- $M_k$ is computed **exactly once** per method per fold, and nothing derived from it flows back anywhere.
- Note the refit: hyperparameters were *selected* using 80 % of the optimization portion per inner fold, but the
  final model is trained on **100 %** of it. This is why §4.3 argued that tuning at 80 % transfers better than
  tuning at 67 %.

## 5.7 Aggregation (descriptive only)

- Mean and standard deviation: $\bar{M} = \frac{1}{5}\sum_k M_k$, $\mathrm{sd}(M_k)$.
- Paired difference against baseline: $\Delta_k = M_k^{PSO} - M_k^{base}$, plus win/tie/loss counts.
- Pooled predictions: the $P_k$ partition $\mathcal{D}$, so every sample gets exactly one test prediction per
  method → one pooled confusion matrix.

**No p-values** (ADR-020). With 5 paired folds the minimum two-sided Wilcoxon p-value is **0.0625**, so
significance is arithmetically impossible — and CV folds are not independent anyway. Claiming significance here
would be dishonest.

## 5.8 Worked example of one particle update

> These numbers are **invented for illustration**, not experimental data. The arithmetic is verified.

Particle *i* at iteration *t*:

| | $n\_estimators$ | $max\_depth$ | $min\_samples\_split$ |
|---|---|---|---|
| position $x$ | 120.4 | 8.2 | 4.6 |
| decoded $\mathrm{dec}(x)$ | 120 | 8 | 5 |
| velocity $v$ | 5.0 | −1.0 | 0.5 |
| personal best $p$ | 130 | 10 | 3 |
| global best $g$ | 180 | 15 | 3 |
| $r_1$ | 0.5 | 0.2 | 0.9 |
| $r_2$ | 0.3 | 0.7 | 0.1 |

With $w = 0.7298$, $c_1 = c_2 = 1.49618$:

| Term | $d = 1$ | $d = 2$ | $d = 3$ |
|---|---|---|---|
| inertia $w v$ | 3.6490 | −0.7298 | 0.3649 |
| cognitive $c_1 r_1 (p - x)$ | 7.1817 | 0.5386 | −2.1545 |
| social $c_2 r_2 (g - x)$ | 26.7517 | 7.1218 | −0.2394 |
| raw $\tilde{v}$ | 37.5824 | 6.9306 | −2.0290 |
| after clamp $\pm(30,\, 3.6,\, 1.6)$ | **30.0** | **3.6** | **−1.6** |
| new position $x + v$ | 150.4 | 11.8 | 3.0 |
| inside bounds? | yes | yes | yes |
| **decoded next configuration** | **150** | **12** | **3** |

Walk through dimension 1 out loud if asked: inertia contributes 3.65; the cognitive term pulls towards 130
giving $1.49618 \times 0.5 \times (130 - 120.4) = 7.18$; the social term pulls hard towards 180 giving
$1.49618 \times 0.3 \times (180 - 120.4) = 26.75$. The raw total 37.58 **exceeds the clamp of 30**, so the step
is capped at 30. The particle moves from (120, 8, 5) to (150, 12, 3) — pulled mainly by the global best, but
**prevented by the clamp from teleporting straight to it**. That is the clamp doing its job.

**Boundary illustration.** If $x_3 = 2.5$ and $v_3 = -1.6$, then $\tilde{x}_3 = 0.9 < l_3 = 2$. The absorbing
wall sets $x_3 = 2$ and $v_3 = 0$, decoding to `min_samples_split = 2`.

## 5.9 Computational budget

Per dataset:

$$\underbrace{5 \times (210_{PSO} + 210_{RS})}_{\text{outer folds}} \;+\; \underbrace{5 \times 1}_{\text{baseline}} \;+\; \underbrace{210}_{\text{deployment}} \;=\; 2{,}315 \text{ evaluations}$$

Across three datasets: **6,945 fitness evaluations**, each one training 5 Random Forests, so roughly **34,725
forest fits**, plus 45 final refits (3 datasets × 5 folds × 3 methods).

---

# PART 6 — THE CLOSED LOOP (THE ACADEMIC CORE)

This is the part the faculty requirement is really about. If you have time for only one section, make it this one.

## 6.1 The loop diagram

```
        ┌──────────────────────────────────────────────────────────────────────┐
        │                                                                      │
        │   ┌─────────────────┐                                                │
        └──▶│   CONTROLLER     │                                               │
            │                  │                                               │
            │      PSO         │                                               │
            │                  │                                               │
            │  pbest / gbest   │                                               │
            │  memory,         │                                               │
            │  velocity and    │                                               │
            │  position update │                                               │
            └────────┬─────────┘                                               │
                     │                                                         │
                     │  MANIPULATED VARIABLE                                   │
                     │  θ = (n_estimators, max_depth, min_samples_split)        │
                     │  (decoded integers)                                     │
                     ▼                                                         │
            ┌──────────────────┐                                               │
            │     PLANT        │                                               │
            │                  │                                               │
            │  Train Random    │                                               │
            │  Forest with θ   │                                               │
            │        +         │                                               │
            │  Inner 5-fold CV │                                               │
            └────────┬─────────┘                                               │
                     │                                                         │
                     │  MEASUREMENT                                            │
                     ▼                                                         │
            ┌──────────────────┐                                               │
            │     SENSOR       │                                               │
            │ mean inner-CV    │                                               │
            │ accuracy = F(θ)  │                                               │
            └────────┬─────────┘                                               │
                     │                                                         │
                     │          FEEDBACK  (this arrow is the whole project)     │
                     └─────────────────────────────────────────────────────────┘

                        repeat for 20 iterations × 10 particles = 210 evaluations

    ═══════════════════════ loop TERMINATES ═══════════════════════

            best θ̂  ──▶  refit RF on 100% of optimization portion
                                        │
                                        ▼
                            HELD-OUT TEST FOLD   ──▶  reported metrics
                            (read ONCE, never fed back — ONE-WAY PATH)
```

## 6.2 The control-systems mapping

| Control concept | This project |
|---|---|
| **Plant / process** | Random Forest training + inner-CV validation |
| **Controller** | PSO |
| **Manipulated variable** | the three hyperparameters $\theta$ |
| **Measured output** | mean inner-CV accuracy |
| **Feedback signal** | that same accuracy, returned as fitness |
| **Controller state / adaptation** | pbest and gbest memory, velocity and position update |
| **One control step** | one particle evaluation |
| **One control cycle** | one swarm iteration (10 evaluations) |
| **Disturbance** | RF's internal randomness (fixed by seed), fold sampling |
| **Open-loop comparator** | random search — same budget, feedback ignored until the end |

## 6.3 Why this is formally a closed loop

Let $\mathcal{H}^t = \{(x_i^s, f_i^s) : s \le t,\; i = 1..N\}$ be the history of proposals and the feedback
received. PSO's next proposal is

$$x^{t+1} = \Phi\!\left(x^t,\, v^t,\, p^t,\, g^t,\, r^t\right)$$

and $p^t, g^t$ are **functions of $\mathcal{H}^t$** — that is, functions of measured feedback.

Therefore **the next control action depends on the measured output of the plant**. That is the definition of a
closed loop. Substituting the definitions, the dependence is explicit:

$$x^{t+1} = x^t + \mathrm{clip}\Big(w v^t + c_1 r_1\big(\underbrace{p^t(\mathcal{H}^t)}_{\text{from feedback}} - x^t\big) + c_2 r_2\big(\underbrace{g^t(\mathcal{H}^t)}_{\text{from feedback}} - x^t\big),\, \pm v_{\max}\Big)$$

Remove the feedback — that is, remove $p$ and $g$ — and PSO degenerates into a random walk. We **test** this:
integration test **IT-02** (the *feedback ablation*) zeroes the $R_1$/$R_2$ contributions and confirms the search
becomes measurably less effective. That test is the empirical proof that the feedback arrow is load-bearing, not
decorative.

## 6.4 The open-loop contrast — random search

This is the scientific control of the experiment, and the reason the comparison is meaningful.

Random search draws $\theta^{(1)}, \dots, \theta^{(M)}$ **i.i.d. uniformly** from $\Omega$, with $M = 210$ — the
*same budget*. Crucially:

$$\theta^{(j+1)} \text{ is independent of } F_k\big(\theta^{(1)}\big), \dots, F_k\big(\theta^{(j)}\big)$$

The feedback is used **only once, at the very end**, to pick the winner:

$$\hat{\theta}^{RS}_k = \theta^{(j^*)}, \qquad j^* = \min \arg\max_j F_k\big(\theta^{(j)}\big)$$

Both methods get:
- the **same budget** (210 evaluations),
- the **same objective** $F_k$,
- the **same inner folds**,
- the **same seed**,
- a **fresh cache each** (so neither can free-ride on the other's work).

**Everything is held constant except the presence of feedback.** Therefore any systematic difference between PSO
and random search is attributable *to the feedback itself*. This is a clean controlled experiment, and it is the
strongest methodological point in the project.

## 6.5 Honest limits — say these before you are asked

This matters. Overclaiming is the fastest way to lose credibility; pre-empting the criticism gains it.

> "To be precise about the framing: this is an **iterative closed optimization loop**, not a physical real-time
> control system. Three honest differences. First, the plant is **static** — a Random Forest has no dynamics,
> no time constant, no transient response; evaluating the same θ twice gives the same answer. Second, there is
> **no setpoint** — I am not tracking a reference value, I am seeking an extremum, so this is closer to
> **extremum-seeking control** than to PID-style regulation. Third, the controller acts through **memory**
> (pbest and gbest) rather than through an error signal $e = r - y$.
>
> What *is* genuinely closed-loop, and what the requirement asks for, is the essential structure: the
> controller's output changes a system parameter, the system responds, the response is measured, and **the next
> output depends on that measurement**. I prove it formally in §6.3, I prove it empirically with the feedback
> ablation test, and I quantify it against the open-loop control."

(Recorded as ADR-022.)

## 6.6 Where the loop is wired — exactly one place

Architecture rule #1 (`CLAUDE.md` §7): PSO and the evaluator are connected **only** in
`src/pso_rf/experiments/runner.py`. There is no disconnected PSO script and no "optimize, then separately
evaluate" shortcut anywhere in the repository.

The adapter that closes the loop is nine lines, and it is worth being able to point at:

```python
# src/pso_rf/experiments/runner.py:65-84
def make_objective(evaluator: FitnessEvaluator) -> Objective:
    """Adapter from the evaluator to the optimizer: only `fitness` is fed back; the rest is recorded."""
    def objective(config: dict[str, int]) -> Evaluation:
        result: FitnessResult = evaluator(config)
        return Evaluation(
            fitness=result.fitness,          # ← the ONLY value PSO sees
            info={                           # ← everything else is logged, never fed back
                "cv_scores": list(result.cv_scores),
                "cv_std": result.cv_std,
                "diag_balanced_accuracy": result.diagnostics.get("balanced_accuracy"),
                "diag_f1_macro": result.diagnostics.get("f1_macro"),
                "cache_hit": result.cache_hit,
                "status": result.status,
                "error": result.error,
                "fit_time_s": result.fit_time_s,
            },
        )
    return objective
```

The `fitness` / `info` split **is** the sensor specification. One scalar goes around the loop; everything else
goes to disk for analysis. Balanced accuracy and macro F1 are computed every single evaluation but are
structurally incapable of influencing the search.

---

# PART 7 — IMPLEMENTATION WALKTHROUGH

## 7.1 The layering rule, and why it is enforced by a test

```
┌─────────────────────────────────────────────────────────────────┐
│  experiments/        ← the ONLY layer allowed to wire the loop   │
│  runner, config, recorder, summary, verify, cli, seeding         │
└───────────────┬─────────────────────────────┬───────────────────┘
                │                             │
                ▼                             ▼
┌───────────────────────────────┐  ┌──────────────────────────────┐
│  optimization/                │  │  evaluation/                 │
│  pso, random_search,          │  │  fitness, splits, final,     │
│  search_space, events         │  │  metrics                     │
│                               │  │                              │
│  NumPy + stdlib ONLY.         │  │  NEVER imports optimization/ │
│  No sklearn. No data.         │  │                              │
│  No dataset names.            │  │                              │
└───────────────────────────────┘  └──────────────┬───────────────┘
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │ datasets/  models/           │
                                   │ preprocessing/  utils/       │
                                   └──────────────────────────────┘
```

**The two rules:**
1. `optimization/` imports only NumPy and the standard library.
2. `evaluation/` never imports `optimization/`.

Unit test **UT-22** (`tests/unit/test_layering.py`) imports every module in `optimization/` and walks its
*transitive* dependency set, asserting that `sklearn`, `scipy`, data objects and dataset names appear nowhere.
This converts an architectural intention into a **mechanically verified property**. If someone adds
`from sklearn...` to the optimizer, the test suite goes red.

Why it matters for the viva: it proves the PSO implementation is genuinely **general-purpose and from scratch**.
It does not know it is tuning a Random Forest. You could point it at any objective function.

## 7.2 File-by-file reference

### `optimization/` — the controller (641 lines)

| File | Lines | Contents |
|---|---|---|
| `pso.py` | 321 | `PSOConfig`, four pure update functions (`velocity_update`, `apply_absorbing_bounds`, `update_personal_best`, `best_index`), `swarm_diversity`, the `PSOOptimizer` class, `_RunState` bookkeeping |
| `search_space.py` | 98 | `IntParam`, `SearchSpace` with `dim`, `size`, `clip`, `decode`, `sample`, `key`, `contains` |
| `events.py` | 122 | `Evaluation`, `EvaluationEvent`, `IterationSummary` (15 fields), `OptimizationResult`, `Callback`, `as_evaluation`, `notify` |
| `random_search.py` | 91 | `RandomSearch` — i.i.d. sampling, no feedback between proposals |
| `__init__.py` | 31 | re-exports |

Design note: the update rules are **pure module-level functions** taking and returning arrays, not methods
mutating state. That is why they can be unit-tested in isolation against hand-computed values — the whole of
`tests/unit/test_pso.py` (288 lines) depends on this choice.

### `evaluation/` — the plant and the sensor (461 lines)

| File | Lines | Contents |
|---|---|---|
| `fitness.py` | 176 | `FitnessEvaluator` (fixed inner folds, caching, parallel folds, failure handling), `FitnessResult` |
| `splits.py` | 149 | `OptimizationPhase`, `HeldOutTestSet`, `OptimizationData`, `FoldData`, `outer_folds`, `full_data`, `TestSetAccessError` |
| `final.py` | 56 | `final_evaluate` — refuses to run inside an `OptimizationPhase` |
| `metrics.py` | 46 | `compute_metrics` — accuracy, balanced accuracy, macro P/R/F1, positive-class metrics, confusion matrix |

Two details worth knowing:

- `OptimizationData` wraps `X`, `y`, `indices` as **read-only NumPy views** (`view.flags.writeable = False`), so
  a bug cannot silently mutate fold data mid-run.
- A failed evaluation returns `fitness = -inf` with `status = "failed"` rather than crashing the run. Since
  pbest updates use strict `>`, a `-inf` can never become a personal best. The run continues; the failure is
  logged and counted in `iterations.csv` as `n_failed`. If *every* evaluation failed, `PSOOptimizer.run` raises
  explicitly rather than returning a meaningless answer.

### `experiments/` — orchestration (1,995 lines)

| File | Lines | Contents |
|---|---|---|
| `config.py` | 746 | 11 frozen dataclasses, YAML loading, deep merge, CLI overrides, **validation that collects all errors at once**, `resolved()`, `config_hash()` |
| `runner.py` | 456 | `make_objective`, `run_fold`, `run_deployment`, `run_experiment`, `boundary_hits`, `_convergence_status` — **the only place the loop is wired** |
| `recorder.py` | 224 | `Recorder` (a `Callback`): buffers evaluation rows, flushes per iteration, writes `evaluations.csv` and `iterations.csv`, prints the per-iteration trace |
| `summary.py` | 211 | builds `summary_folds.csv` and `summary.csv` from saved `final.json` files only |
| `verify.py` | 195 | `verify_results` (completeness, test coverage, summary recomputation, **log temporal ordering**), `compare_runs` |
| `cli.py` | 135 | subcommands `run`, `plot`, `verify`, `compare`, `web`, `demo`, `audit` |
| `seeding.py` | 28 | `RunSeeds`, `make_rng` — one `Generator` per optimizer, never global state |

`config.py` is the largest file in the project, which is deliberate: every magic number lives in YAML, and the
validator reports **all** problems in one pass instead of failing on the first one.

### `datasets/`, `models/`, `preprocessing/`, `utils/`, `visualization/`

| File | Lines | Contents |
|---|---|---|
| `datasets/audit.py` | 191 | `describe`, `audit`, `select_fitness_metric` (the class-ratio gate), `count_duplicate_rows`, markdown rendering |
| `datasets/heart_cleveland.py` | 79 | reads `processed.cleveland.data`, **verifies SHA-256** against `MANIFEST.json`, `?` → NaN, `num > 0 → 1` |
| `datasets/base.py` | 42 | `DatasetBundle` frozen dataclass with shape and label validation |
| `datasets/iris.py` / `digits.py` | 29 / 32 | thin wrappers over scikit-learn loaders |
| `models/random_forest.py` | 33 | `build_model` → `Pipeline([imputer?] + [RandomForestClassifier(...)])` |
| `models/baseline.py` | 13 | `BASELINE_CONFIG` = (100, None, 2) as an immutable `MappingProxyType` |
| `preprocessing/pipeline.py` | 32 | `PreprocessingSpec`, `build_steps` |
| `utils/io.py` | 93 | **atomic** JSON writes (temp file + `os.replace`), append-only CSV with header validation |
| `utils/hashing.py` | 37 | `sha256_file`, `sha256_arrays`, `canonical_json`, `sha256_json` |
| `utils/manifest.py` | 84 | package versions, **git commit + dirty flag**, platform, CPU count, dataset checksums |
| `utils/log.py` | 87 | console at INFO, file at DEBUG with UTC timestamps, `RunLoggerAdapter` prefixing `[dataset\|fold k\|method\|seed s]` |
| `visualization/plots.py` | 407 | figures F3–F11, all read from saved files, each recording its inputs in `SOURCES.json` |

### Frontends — `app/` (Streamlit, 1,661 lines) and `web/` (Tornado, 757 lines)

Both exist for the Review 2 live demonstration (ADR-026, ADR-027). The architectural rule that keeps them
honest: **live runs go through the real `run_fold` with observer callbacks only; the results and replay pages
read saved files. Neither frontend has its own copy of the loop.** A frontend callback can watch the search but
cannot alter it.

`app/explain.py` is worth knowing about for the demo: it runs the **test-isolation proof live in the browser** —
it actually calls `HeldOutTestSet.reveal()` inside an `OptimizationPhase` and displays the resulting
`TestSetAccessError`. It also runs the feedback ablation live against a stub objective with a known optimum.

## 7.3 Reproducibility machinery

| Mechanism | Implementation |
|---|---|
| **Seed policy** | Outer split 42 (fixed). Run seed $s_k = k$ for $k \in \{0..4\}$ drives the PSO generator, the random-search generator, the inner-CV shuffle *and* RF `random_state`. Deployment seed 5. |
| **No global random state** | NumPy `Generator` objects only. `np.random.seed`, `np.random.rand` and the `random` module are forbidden project-wide. |
| **Fixed RNG draw order** | positions → velocities → ($R_1$, $R_2$) per iteration. Documented in the `pso.py` module docstring, because changing the order changes the run. |
| **Config hash** | SHA-256 of the canonical JSON of the resolved config: `80474a0c3a1485fe5ca85ffee0ff948c9eb261e251fcd2156a1eef7a54143ba5` |
| **Git provenance** | `manifest.json` records commit `8bd59275d345df3f648d114ada19f44b4c264f95` and `"git_dirty": false` |
| **Dataset checksums** | Heart raw file SHA-256 `a74b7efa...`, plus `sha256_arrays` for all three datasets |
| **Pinned dependencies** | numpy 1.26.4, scikit-learn 1.7.2, pandas 2.3.3, matplotlib 3.10.6, PyYAML 6.0.1, joblib 1.5.2, scipy 1.14.1 |
| **Atomic writes** | temp file + `os.replace`, so a crash cannot leave a half-written JSON |
| **Re-run equality** | IT-08 re-runs a config and asserts every result file is identical, excluding timing fields |

`"git_dirty": false` is a small but strong claim: the experiment of record was produced from a **clean,
committed tree**, so the exact source that generated these numbers is recoverable from commit `8bd5927`.

---

# PART 8 — EXPERIMENTAL PROTOCOL

## 8.1 The run matrix

```
3 datasets  ×  5 outer folds  ×  3 methods  =  45 runs
                            +  3 deployment PSO runs
                            =  48 runs total
```

| Dimension | Values |
|---|---|
| Datasets | iris, digits, heart_cleveland |
| Outer folds | 0, 1, 2, 3, 4 (fold *k* is run *k*'s held-out test fold) |
| Methods | baseline, random_search, pso |
| Budget | PSO 210 = RS 210; baseline 1 |

## 8.2 The three datasets

| Property | Iris | Digits | Heart Disease (Cleveland) |
|---|---|---|---|
| Source | `sklearn.datasets.load_iris` | `sklearn.datasets.load_digits` | UCI id 45, `processed.cleveland.data`, vendored with SHA-256 |
| Samples | 150 | 1,797 | 303 |
| Features | 4 | 64 | 13 |
| Classes | 3 | 10 | 2 |
| Class counts | 50 / 50 / 50 | 174 – 183 | 164 / 139 |
| Class ratio | 1.000 | 1.052 | 1.180 |
| Missing values | none | none | `ca` 4, `thal` 2 |
| Duplicate rows | **1** (kept) | 0 | 0 |
| Preprocessing | none | none | `SimpleImputer(most_frequent)` in the pipeline |
| Test fold size | 30 | ~359 | 61 |
| Optimization size | 120 | 1,437 | 242 |
| Inner fold size | 24 | ~287 | ~48 |
| Role in the study | small, easy, nearly saturated | larger, multi-class, image-like | small, noisy, real clinical data with missing values |

Source: `results/20260925-083324_default/*/audit.json` and `manifest.json`.

Why these three (ADR-021): they are the proposal's datasets, and they deliberately span three regimes — a tiny
easy set, a larger 10-class set, and a small noisy real-world clinical set. If the conclusions hold across all
three, they generalize; if they differ, the difference is itself a finding (and it is — see §10.5).

**Heart Disease specifics (ADR-009):** the UCI "Heart Disease" dataset has four sites; we use the **Cleveland**
subset, which is the one with 303 complete-ish rows and the standard 13-feature processed form. The target
column `num` is 0–4 (severity); we binarize with `num > 0 → 1`, giving 164 no-disease / 139 disease. Missing
values are encoded `?` in the raw file and converted to NaN. The file is **committed to the repository** with
its SHA-256 recorded, so the data cannot silently change under the experiment.

**The Iris duplicate row** was found by the audit and **deliberately kept**: removing it would deviate from the
canonical dataset everyone else reports on. The audit code *would* raise `NotImplementedError` if duplicates
exceeded 1 % of a dataset, but 1/150 = 0.67 % is below that, so the guard does not fire. Recording the fact
rather than quietly dropping the row is the point.

## 8.3 Seeds

| Seed | Value | Controls |
|---|---|---|
| Outer split | 42 | `StratifiedKFold(5, shuffle=True, random_state=42)` — same folds for all methods and all datasets |
| Run seed $s_k$ | $k$ (0–4) | PSO generator, RS generator, inner-CV shuffle, RF `random_state` |
| Deployment | 5 | the full-data PSO run |

Using the *same* run seed for all three methods on a given fold is what makes the comparison **paired** — PSO,
random search and the baseline all face identical inner folds and identical RF randomness. The only difference
is the search strategy.

## 8.4 The deployment run

After the five outer folds, PSO runs **once more on the entire dataset** (seed 5) to produce the configuration
you would actually ship.

It has **no test score**, by construction — there is no held-out data left. Its expected performance is
**estimated by the outer-CV mean of PSO**, and `deployment.json` carries this note verbatim:

> "No held-out test data exists for the deployment run; performance is estimated by the outer 5-fold CV of the
> same procedure. validation_fitness is optimistically biased."

That sentence is in the output file, not just the report. The code refuses to let you misread the number.

## 8.5 Measured runtime

| Dataset | baseline | random_search | pso |
|---|---|---|---|
| Iris | 3.78 s | 372.5 s | 419.0 s |
| Digits | 8.36 s | 989.1 s | 900.3 s |
| Heart | 3.03 s | 365.7 s | 320.8 s |

(`summary.csv`, column `total_time_s_sum`, summed over the 5 folds.)

**Whole experiment: 62.7 minutes** (`manifest.json`: started 08:33:24 Z, finished 09:36:06 Z, on Windows 11
build 26200, Python 3.10.11, 20 CPUs).

The pre-run projection was ≤ 37.5 min. The honest explanation, recorded in the ADR-005 amendment: the benchmark
used short, warm, repeated calls and underestimated **sustained** cost by about 2.7×, most likely because of
per-call parallel dispatch overhead and sustained-load CPU clocking — though that cause is *not proven*, and the
ADR says so. The 2-hour fallback threshold was never reached, so the 5-fold decision stands.

Note that PSO is **faster than random search on Digits and Heart** but slower on Iris. The reason is the cache:
PSO revisits configurations as it converges, and a cache hit costs nothing. See §9.6.

---

# PART 9 — RESULTS — EVERY NUMBER

Source: `results/20260925-083324_default/summary.csv` and `summary_folds.csv`.

## 9.1 Headline table — held-out test accuracy (mean ± sd over 5 folds)

| Dataset | Default RF | Random Search | **PSO** |
|---|---|---|---|
| **Iris** | 0.9600 ± 0.0435 | 0.9533 ± 0.0380 | **0.9600 ± 0.0435** |
| **Digits** | 0.9722 ± 0.0081 | 0.9750 ± 0.0028 | **0.9744 ± 0.0046** |
| **Heart Cleveland** | 0.8316 ± 0.0359 | 0.8251 ± 0.0412 | **0.8283 ± 0.0478** |

## 9.2 Validation fitness — the objective PSO actually optimizes

| Dataset | Default RF | Random Search | **PSO** |
|---|---|---|---|
| **Iris** | 0.9483 | 0.9583 | **0.9583** |
| **Digits** | 0.9713 | 0.9755120 | **0.9755139** |
| **Heart Cleveland** | 0.8126 | 0.8414286 | **0.8414626** |

**This is the single most important table in the project, and you must present it alongside §9.1.**

PSO's validation fitness is **≥ random search on all three datasets, and strictly > the default RF on all
three** (Iris +1.00, Digits +0.42, Heart +2.88 percentage points). The optimizer **works**: it does what it was
asked to do, which is maximize $F_k$.

But compare with §9.1: those validation gains **do not carry over to test accuracy**. That gap *is* the result.
Discussion in §10.1.

## 9.3 Win / tie / loss over all 15 folds

| Comparison | Wins | Ties | Losses |
|---|---|---|---|
| PSO vs default RF | **4** | **6** | **5** |
| PSO vs random search | **4** | **8** | **3** |
| Random search vs default RF | 3 | 9 | 3 |

Per dataset (`summary.csv`, columns `wins`/`ties`/`losses` and `wins_vs_rs`/`ties_vs_rs`/`losses_vs_rs`):

| Dataset | PSO vs baseline | PSO vs random search |
|---|---|---|
| Iris | 0 W / 5 T / 0 L | 1 W / 4 T / 0 L |
| Digits | 2 W / 0 T / 3 L | 1 W / 2 T / 2 L |
| Heart | 2 W / 1 T / 2 L | 2 W / 2 T / 1 L |

Reading: **4–6–5 against the default RF is a tie.** PSO neither reliably helps nor reliably hurts. Against
random search, 4–8–3 leans very slightly towards PSO but is well inside noise with 15 paired folds.

Remember (§5.7): **no p-values**, because with 5 paired folds the minimum two-sided Wilcoxon p-value is 0.0625.
Significance cannot be established, and claiming it would be dishonest.

## 9.4 Per-fold detail

### Iris — the plateau case

| Fold | Baseline acc | RS acc | **PSO acc** | PSO config (trees, depth, split) | PSO val fitness | $t_c$ |
|---|---|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | **1.0000** | (179, 3, 8) | 0.9583 | 0 |
| 1 | 0.9667 | 0.9667 | **0.9667** | (174, 9, 6) | 0.9417 | 0 |
| 2 | 0.9333 | 0.9333 | **0.9333** | (89, 7, 9) | 0.9750 | 0 |
| 3 | 1.0000 | 0.9667 | **1.0000** | (67, 2, 8) | 0.9500 | 4 |
| 4 | 0.9000 | 0.9000 | **0.9000** | (191, 14, 3) | 0.9667 | 0 |

PSO matched the baseline on **every** fold, and beat random search on fold 3 (1.0000 vs 0.9667). Note
$t_c = 0$ on four of five folds: **the best configuration was already in the initial random swarm.** On a
30-sample test fold each error is worth 3.33 accuracy points, so this dataset has almost no resolution to
distinguish methods. Iris is saturated — and that is a legitimate, reportable finding, not a failure.

### Digits — where the depth bound bites

| Fold | Baseline acc | RS acc | **PSO acc** | PSO config | PSO val fitness | $t_c$ |
|---|---|---|---|---|---|---|
| 0 | 0.9694 | 0.9778 | **0.9778** | (152, 19, 2) | 0.9729 | 3 |
| 1 | 0.9750 | 0.9750 | **0.9722** | (178, 13, 5) | 0.9777 | 8 |
| 2 | 0.9610 | 0.9721 | **0.9721** | (172, 11, 3) | 0.9757 | 5 |
| 3 | 0.9833 | 0.9777 | **0.9805** | (153, 9, 6) | 0.9736 | 4 |
| 4 | 0.9721 | 0.9721 | **0.9694** | (107, 14, 6) | 0.9778 | 8 |

Both tuned methods beat the baseline on folds 0 and 2 (by 0.83 and 1.11 points) but lose on folds 3 and 4.
Observe fold 0: PSO chose `max_depth = 19`, right against the upper bound of 20. This is consistent with the
risk flagged in ADR-003 **before the run**: the default RF grows **unlimited-depth** trees on Digits, and our
cap of 20 may be binding. So part of the Digits result is a **search-space limitation**, not an optimizer
limitation. Being able to say "I predicted this in advance and recorded it" is much stronger than discovering it
afterwards.

### Heart Cleveland — the highest variance, and the biggest single win

| Fold | Baseline acc | RS acc | **PSO acc** | PSO config | PSO val fitness | $t_c$ |
|---|---|---|---|---|---|---|
| 0 | 0.8689 | 0.8852 | **0.9016** | (200, 6, 7) | 0.8222 | 1 |
| 1 | 0.7869 | 0.7869 | **0.8033** | (117, 4, 5) | 0.8473 | 6 |
| 2 | 0.8525 | 0.8033 | **0.8033** | (106, 2, 7) | 0.8425 | 16 |
| 3 | 0.8000 | 0.8000 | **0.7833** | (188, 14, 6) | 0.8474 | 5 |
| 4 | 0.8500 | 0.8500 | **0.8500** | (113, 3, 8) | 0.8479 | 4 |

The spread is the story. **Fold 0: PSO +3.28 points** over the baseline (0.9016 vs 0.8689) — the best single
result in the project. **Fold 2: PSO −4.92 points** (0.8033 vs 0.8525) — the worst. With only 61 test samples,
one sample is worth 1.64 accuracy points, so a 3-sample swing is 4.9 points. **Most of this spread is sampling
noise, not optimizer skill.** Say that out loud before anyone else does.

Fold 0 is also the boundary-hit case: PSO chose `n_estimators = 200`, exactly the upper bound, and `final.json`
records `"boundary_hits": ["n_estimators"]`. This is the absorbing-wall design (§3.5) paying off — a reflecting
boundary would have pushed the particle away from the point it wanted.

Fold 2 has $t_c = 16$ — gbest was still improving late. Combined with its loss on test, this is the clearest
single instance of **overfitting the validation signal**: PSO kept finding configurations with better $F_k$, and
those gains were noise that did not transfer.

## 9.5 Deployment configurations

| Dataset | Recommended (trees, depth, split) | Validation fitness | Performance estimate | $t_c$ |
|---|---|---|---|---|
| Iris | **(171, 17, 6)** | 0.9600 | 0.9600 ± 0.0435 | 0 |
| Digits | **(178, 18, 3)** | 0.9783 | 0.9744 ± 0.0046 | 17 |
| Heart | **(110, 3, 4)** | 0.8450 | 0.8283 ± 0.0478 | 4 |

(`*/deployment/pso/deployment.json`; the performance estimate is PSO's outer-CV mean from `summary.csv`.)

The Digits deployment run has $t_c = 17$ out of 20 — still improving when the budget ran out, suggesting Digits
would benefit from more iterations. Heart's recommendation, `max_depth = 3`, is a **strongly regularized**
forest, which is exactly what you would expect for 303 noisy clinical rows.

## 9.6 Cache behaviour — quantitative evidence of convergence

`n_unique_fits` counts distinct configurations actually trained out of 210 evaluations.

| Dataset | Random search | **PSO** | PSO repeat rate |
|---|---|---|---|
| Iris | 209.6 | **188.8** | 10.1 % |
| Digits | 209.6 | **159.0** | 24.3 % |
| Heart | 209.6 | **164.2** | 21.8 % |

This table is a lovely, often-overlooked piece of evidence. Random search touches ~210 distinct points because
i.i.d. draws from 25,821 points rarely collide. **PSO revisits 10–24 % of its evaluations**, because particles
converge and multiple continuous positions decode to the same integer configuration.

Two conclusions:
1. It is **direct, measured evidence that the swarm contracts** — exactly what the diversity curve (F11) shows
   graphically.
2. It explains why PSO was **faster than random search on Digits and Heart**: cache hits are free. PSO did more
   effective search per second.

## 9.7 Convergence summary

| Dataset | Convergence iteration $t_c$ per fold | Interpretation |
|---|---|---|
| Iris | 0, 0, 0, 4, 0 | immediate — plateau, best often in the initial swarm |
| Digits | 3, 8, 5, 4, 8 | steady mid-run improvement |
| Heart | 1, 6, 16, 5, 4 | variable; fold 2 still improving late |

With $T = 20$ and most $t_c$ well below 20, the budget was **adequate** on Iris and Heart. Digits (deployment
$t_c = 17$, folds at 8) is the one case where more iterations would plausibly have helped.

## 9.8 Pooled test accuracy

Because the five outer folds partition the dataset, every sample gets exactly one test prediction per method,
giving a single pooled confusion matrix (figure **F10**).

| Dataset | Baseline | Random search | **PSO** |
|---|---|---|---|
| Iris | 0.9600 | 0.9533 | **0.9600** |
| Digits | 0.9722 | 0.9750 | **0.9744** |
| Heart | 0.8317 | 0.8251 | **0.8284** |

Example pooled-level detail, Heart fold 0 PSO (`final.json`):
confusion matrix $\begin{pmatrix} 31 & 2 \\ 4 & 24 \end{pmatrix}$ — 2 false positives, 4 false negatives;
positive-class precision 0.923, recall 0.857, F1 0.889. For a clinical screening task the 4 false negatives
(missed disease) matter more than the 2 false positives, which is worth mentioning as a limitation of optimizing
plain accuracy.

## 9.9 The nine figures

Generated by `python -m pso_rf plot`, all from saved files only, each recording its inputs in
`plots/<exp_id>/SOURCES.json`:

| Figure | Content | What to say about it |
|---|---|---|
| **F3** | gbest validation accuracy vs iteration, per fold + mean | The convergence curve. Monotone non-decreasing by construction (§3.8). |
| **F4** | best-so-far vs evaluation count, PSO vs RS | **The anytime curve — the single best slide for the closed-loop argument.** Same budget, same objective; any separation is the feedback. |
| **F5** | held-out test accuracy bars + per-fold dots | Shows means *and* spread. The dots are why we do not claim significance. |
| **F6** | decoded hyperparameter values per particle vs iteration (fold 0) | Particles visibly converging in all 3 dimensions. |
| **F7** | particle fitness scatter + swarm mean + gbest per iteration | Swarm mean rising towards gbest = the swarm learning. |
| **F8** | gbest hyperparameter steps vs iteration, all folds | gbest moves in discrete jumps (strict `>` + decode). |
| **F9** | per-fold test accuracy deltas vs default RF | The honest 4–6–5 picture, fold by fold. |
| **F10** | pooled out-of-fold confusion matrices, baseline vs PSO | Where the errors actually are. |
| **F11** | swarm diversity (normalized) vs iteration | **Exploration → exploitation, measured.** Pair it with §9.6. |

---

# PART 10 — ANSWERS TO THE RESEARCH QUESTIONS

Answer these in exactly this spirit: direct, quantified, and honest about the null results.

## 10.1 RQ1 — Does PSO-tuned RF improve or maintain test performance vs the default RF?

**Answer: it maintains it. It does not reliably improve it. 4 wins, 6 ties, 5 losses over 15 folds.**

| Dataset | Mean $\Delta$ accuracy vs baseline | sd of $\Delta$ |
|---|---|---|
| Iris | **0.0000** | 0.0000 |
| Digits | **+0.0022** | 0.0069 |
| Heart | **−0.0033** | 0.0316 |

The mean deltas are all smaller than their own standard deviations. This is a **null result**, and it is
reported as one (ADR-020, rule 5).

**Why, mechanistically?** Three reasons, in order of importance:

1. **The default RF is a strong, well-chosen baseline.** scikit-learn's defaults (100 trees, unlimited depth,
   `min_samples_split=2`) are good *because* the library authors tuned them against many datasets. Beating them
   on three classic, clean benchmarks is genuinely hard.
2. **These datasets are near their accuracy ceiling.** Iris at 96 %, Digits at 97 %. The remaining errors are
   largely irreducible (overlapping classes, ambiguous digits), so **no hyperparameter choice can fix them**.
3. **Our search space excludes the most influential knob.** `max_features` controls the inter-tree correlation
   $\rho$, which §2.1 showed sets the *floor* of the ensemble variance. We do not tune it, because the project
   proposal fixed the three decision variables. ADR-003 recorded this limitation before the run.

**And the crucial nuance — §9.2.** PSO *did* maximize its objective: its validation fitness beat the default RF
on all three datasets (Iris +1.00, Digits +0.42, Heart +2.88 points). The optimizer is not broken. The gain
simply did not transfer to held-out data. That is the **generalization gap / winner's curse**, and demonstrating
it cleanly is a more valuable outcome than a lucky win would have been.

## 10.2 RQ2 — Does PSO's feedback help, compared with the same budget without feedback?

**Answer: marginally and inconsistently on test accuracy (4 W / 8 T / 3 L); consistently but minutely on the
objective itself; and clearly on search efficiency.**

Three separate measurements, which must not be conflated:

| Measurement | Result | Verdict |
|---|---|---|
| Held-out test accuracy | 4 W / 8 T / 3 L over 15 folds | leans to PSO, inside noise |
| Validation fitness (the actual objective) | PSO ≥ RS on **all three** datasets | PSO wins, but by a tiny margin |
| Search efficiency | PSO used 159–189 unique fits vs RS's ~210 | PSO reached equal-or-better fitness with **10–24 % fewer distinct model trainings** |

The validation-fitness margins are startlingly small: Iris identical at 0.9583; Digits 0.9755139 vs 0.9755120;
Heart 0.8414626 vs 0.8414286. **Both methods are hitting the same ceiling of $F_k$.**

This is the most intellectually interesting finding, and here is the explanation to give:

> With 210 evaluations over 25,821 configurations, and a fitness surface that is **flat near the optimum**,
> random search is a genuinely strong competitor. The flatness is the key: when many configurations tie at the
> top, you do not need a clever trajectory to find one of them — you just need to sample anywhere in a large
> good region. Feedback buys you very little when the landscape has no structure to exploit.
>
> That is a real, publishable insight about **when** closed-loop optimization pays off. It pays when the
> landscape is rugged or high-dimensional, where blind sampling is hopeless. It pays little on a flat,
> low-dimensional, 3-parameter box like this one. I can demonstrate the loop works — the feedback ablation test
> IT-02 shows removing feedback degrades the search — while being honest that on *this* problem the loop's
> advantage is small.

## 10.3 RQ3 — How does the PSO search converge?

**Answer: fast and early, then flat — consistent with a plateau-rich surface.**

- Convergence iterations: Iris {0,0,0,4,0}, Digits {3,8,5,4,8}, Heart {1,6,16,5,4}.
- On Iris, four of five folds had $t_c = 0$: **the best configuration was in the initial random swarm.** That is
  not a PSO failure, it is a statement about the surface — on a near-flat top, ten random draws already land on a
  tie-for-best.
- Diversity (F11) falls monotonically on all runs: the swarm contracts, exploration gives way to exploitation.
- The 10–24 % cache-hit rate (§9.6) independently confirms contraction: particles are revisiting the same decoded
  configurations.
- gbest is **non-decreasing by construction** (strict `>`, §3.8), so F3 is a valid convergence plot.
- Of all runs, only Digits (deployment $t_c = 17$, folds reaching 8) shows signs of being budget-limited.

## 10.4 RQ4 — How stable are the results across folds and seeds?

**Answer: the test metrics are stable; the *chosen configurations* are not.**

Test accuracy standard deviations:

| Dataset | Baseline sd | RS sd | PSO sd |
|---|---|---|---|
| Iris | 0.0435 | 0.0380 | 0.0435 |
| Digits | 0.0081 | 0.0028 | 0.0046 |
| Heart | 0.0359 | 0.0412 | 0.0478 |

Digits is the most stable (large test folds, ~359 samples). Iris and Heart are much noisier because 30 and 61
test samples mean one sample is worth 3.33 and 1.64 accuracy points respectively.

The **selected hyperparameters**, however, vary enormously:

| Dataset | `n_estimators` chosen across folds | `max_depth` chosen | `min_samples_split` chosen |
|---|---|---|---|
| Iris | 179, 174, 89, 67, 191 | 3, 9, 7, 2, 14 | 8, 6, 9, 8, 3 |
| Digits | 152, 178, 172, 153, 107 | 19, 13, 11, 9, 14 | 2, 5, 3, 6, 6 |
| Heart | 200, 117, 106, 188, 113 | 6, 4, 2, 14, 3 | 7, 5, 7, 6, 8 |

**This is the most informative instability in the project**, and it is worth a slide of its own. On Iris,
`max_depth` ranges from 2 to 14 across folds while test accuracy is **identical on every fold**. On Heart,
`n_estimators` ranges from 106 to 200.

Interpretation: **the fitness surface has a large flat optimal region, so many very different configurations are
equally good.** The optimizer is not unstable — the *problem* is underdetermined. This also predicts the ADR-003
expectation partially wrong: `n_estimators` did **not** uniformly drift to 200 (only 1 of 15 folds hit the
bound), which tells us accuracy had already saturated well below 200 trees. That is a genuine measured finding
that refines our own prior expectation.

## 10.5 RQ5 — How do the answers differ across datasets?

**Answer: yes, and the differences follow directly from dataset size and difficulty.**

| | Iris (150, easy) | Digits (1797, multi-class) | Heart (303, noisy clinical) |
|---|---|---|---|
| Baseline accuracy | 0.9600 | 0.9722 | 0.8316 |
| PSO vs baseline | 0 W / 5 T / 0 L | 2 W / 0 T / 3 L | 2 W / 1 T / 2 L |
| Validation gain over default | +1.00 pt | +0.42 pt | **+2.88 pt** |
| Test $\Delta$ sd | 0.0000 | 0.0069 | **0.0316** |
| Convergence | immediate ($t_c$ mostly 0) | mid-run (3–8) | variable (1–16) |
| Cache repeat rate | 10.1 % | 24.3 % | 21.8 % |
| Limiting factor | **saturation** — no headroom | **the `max_depth ≤ 20` bound** | **noise** — 61 test samples |

The three datasets fail to benefit for **three different reasons**, and being able to name each one is what turns
a null result into an analysis:

- **Iris** is saturated. At 96 % on 30 test samples there is nothing to win; PSO correctly finds the plateau and
  ties.
- **Digits** is partly bound-limited. The default grows unlimited-depth trees; our cap is 20, and PSO pushed to
  19 on fold 0. The search space, not the search, is the constraint.
- **Heart** has the **most headroom in validation fitness (+2.88 points)** and the **largest test variance**.
  This is the clearest case of the generalization gap: the most real validation improvement, and it still did not
  transfer, because 61 test samples cannot resolve a 1-point difference.

Headline cross-dataset conclusion: **the benefit of closed-loop hyperparameter optimization depends on whether
there is headroom to find, and whether the validation set is large enough for the headroom to be real rather
than noise.** On these three datasets, at least one of those conditions fails each time.

---

# PART 11 — VERIFICATION, TESTS AND PROOFS

## 11.1 Test suite status

```
203 passed in 202.51s (3 min 22 s)
```

| Category | Files | Lines | Coverage |
|---|---|---|---|
| Unit (`tests/unit/`) | 14 | ~1,900 | utils, layering, audit, datasets, preprocessing, models, fitness, splits, metrics, search space, PSO, random search, recorder, config |
| Integration (`tests/integration/`) | 6 | ~390 | isolation proofs, closed-loop proofs, parallel equivalence |
| Experiment (`tests/experiment/`) | 1 | 163 | end-to-end run, completeness, summary consistency, timing order, reproducibility |
| App (`tests/app/`) | 2 | 258 | Streamlit data/charts, Tornado endpoints |
| **Total** | **27** | **~2,720** | |

Tests run on `configs/test.yaml` (4 particles × 2 iterations, 3 inner folds) so the full suite stays under 5
minutes.

## 11.2 The isolation proofs — "prove you didn't cheat"

These are the tests to name if asked how you know there is no leakage. All pass.

| ID | File | What it proves |
|---|---|---|
| **UT-15** | `test_splits.py` | `HeldOutTestSet.reveal()` **raises** `TestSetAccessError` inside an `OptimizationPhase` |
| **UT-22** | `test_layering.py` | `optimization/` has **no** transitive dependency on sklearn, scipy, data objects or dataset names |
| **UT-20** | `test_preprocessing.py` | the imputer's statistics are learned from **training rows only** |
| **IT-04** | `test_isolation_indices.py` | **zero index overlap** between optimization and test portions, on all 3 real datasets |
| **IT-05** | `test_isolation_behaviour.py` | a **spy** on the test arrays records no reads during optimization |
| **IT-06** | `test_isolation_behaviour.py` | **corrupting every test label** leaves the PSO trajectory bit-for-bit identical |
| **IT-07** | `test_isolation_behaviour.py` | **corrupting every test feature** leaves the PSO trajectory bit-for-bit identical |
| **ET-04** | `test_experiment.py` | `optimization_finished_at` **precedes** `test_evaluated_at` in every `final.json` |

IT-06 and IT-07 are the strongest of these, and the ones to describe in a viva:

> "I take a fold, scramble every single label in the held-out test set, re-run the entire PSO optimization, and
> assert the trajectory is **bit-for-bit identical** — same positions, same velocities, same fitness values, same
> gbest. If even one byte of test data influenced the search, those runs would diverge. That is not a code
> review; it is a mechanical proof of independence."

## 11.3 The closed-loop proofs

| ID | File | What it proves |
|---|---|---|
| **IT-01** | `test_closed_loop_stub.py` | the loop **converges** on a stub objective with a known optimum (no ML involved) |
| **IT-02** | `test_closed_loop_stub.py` | **feedback ablation** — removing the feedback contribution measurably degrades the search |
| **IT-03** | `test_closed_loop_rf.py` | the **real** RF closed loop runs end-to-end through `run_fold` on real Iris and Heart data |

IT-02 is the empirical counterpart to the formal argument in §6.3. IT-01 is important for a different reason:
testing the optimizer against a **stub objective with a known optimum** proves PSO is correct *independently* of
the Random Forest. If a result looks wrong, you can tell whether the optimizer or the plant is at fault.

## 11.4 Independent verification of the saved results

```
$ python -m pso_rf verify --results results/20260925-083324_default
OK: all checks passed
```

`verify_results` (`src/pso_rf/experiments/verify.py`) re-reads the result directory from disk and checks:

1. **Completeness** — every dataset × fold × method has `final.json`, `predictions.csv`, `evaluations.csv` with
   the **correct row count**, and `iterations.csv` for PSO.
2. **Test coverage** — pooling all folds' predictions covers every sample **exactly once** (proving the outer
   folds truly partition the data).
3. **Summary consistency** — `summary.csv` and `summary_folds.csv` are **recomputed from the raw `final.json`
   files** and must match byte-for-byte. You cannot hand-edit a summary and get away with it.
4. **Temporal isolation** — the whole `run.log` is parsed and every "test set revealed" line must come **after**
   the matching "optimization phase closed" line.

Check 3 is the one that enforces `CLAUDE.md` rule "No fake results": the reported numbers are provably derived
from the per-run artifacts.

## 11.5 Acceptance criteria — all met

| Criterion | Status | Evidence |
|---|---|---|
| All UT/IT/ET tests pass; isolation + closed-loop proofs pass | ✅ | 203/203 |
| `python -m pso_rf run --config configs/default.yaml` on a clean tree | ✅ | `manifest.json`: `"git_dirty": false`, `"status": "completed"` |
| Every dataset × fold × method has complete artifacts (ET-02) | ✅ | `verify`: OK |
| Summaries recomputable exactly (ET-03) | ✅ | `verify`: OK |
| Re-run reproduces files exactly, excluding timing (IT-08) | ✅ | test passes |
| Figures F3–F11 generated from files with `SOURCES.json` | ✅ | 9 PNGs in `plots/` |
| Report and ≤10-slide deck answer RQ1–RQ5 truthfully | ✅ | Part 10 |
| Review 1 and Review 2 mappings evidenced | ✅ | `docs/REVIEW_*_MAPPING.md` |
| Docs reflect the implemented system | ✅ | 27 ADRs, 14 documents |

## 11.6 The two intentional `NotImplementedError`s

Be ready for this, because a sharp examiner will grep for them.

1. **`optimization/pso.py:122`** — raises for any `boundary`, `topology` or `update` other than
   `absorb`/`gbest`/`synchronous`. This is a **guard, not a gap**: reflecting boundaries and ring topology are
   documented extension points (ADR-012, ADR-014) that were never validated, so the code refuses to run them
   rather than silently producing results from an untested path. The *implemented* variant is the one the whole
   design specifies.
2. **`datasets/audit.py:69`** — raises if exact duplicate rows exceed 1 % of a dataset, because DR-005 requires
   deduplication before splitting in that case. Measured duplicates: Iris 1/150 = 0.67 %, Digits 0, Heart 0. The
   guard never fires. It exists so a future dataset cannot slip past the rule unnoticed.

Neither is an incomplete feature. Both are **fail-loud guards** on paths the design deliberately excludes.

---

# PART 12 — SLIDE-BY-SLIDE PRESENTATION SCRIPT

Target: ≤ 10 slides. Roughly 60–90 seconds per slide.

### Slide 1 — Title
PSO-Based Random Forest Hyperparameter Optimization · Anush Ramesh · CB.EN.U4ELC23005 · EEE, Amrita
Coimbatore · Evolutionary Optimization.

*Say:* "I built PSO from scratch and used it as a controller in a closed feedback loop that tunes a Random
Forest."

### Slide 2 — The problem
25,821 configurations. 0.354–1.000 s per evaluation. Exhaustive search ≈ 180 hours. No gradient exists.

*Say:* "Hyperparameters are a design choice, not something learned. Accuracy is a step function of them, so
there is no gradient to follow. The space has 25,821 valid points and each one costs a second to evaluate. That
is a derivative-free, expensive, black-box optimization problem — which is exactly what swarm methods are for."

### Slide 3 — Block diagram (THE key slide)
The loop from §6.1, plus the control mapping table, plus the one-way test lane drawn **outside** the loop.

*Say:* "PSO is the controller. The plant is Random Forest training plus inner cross-validation. The manipulated
variables are three hyperparameters. The measurement is mean inner-CV accuracy, and it is fed straight back as
fitness. Twenty iterations, ten particles, 210 evaluations. The held-out test fold is **outside** the loop
entirely — it is read once, after the loop terminates, and nothing derived from it ever flows back."

Point at the feedback arrow when you say "fed straight back". That arrow is the requirement.

### Slide 4 — Objective, decision variables, constraints
$\theta = (n\_estimators, max\_depth, min\_samples\_split)$; bounds (50–200, 2–20, 2–10); $|\Omega| = 25{,}821$;
maximize $F_k(\theta)$ = mean 5-fold inner-CV accuracy.

*Say:* "Maximize the mean accuracy over five stratified inner folds, which are drawn once and **frozen for the
whole run** — so every candidate is judged on identical data, which makes the fitness deterministic and the
comparison between particles fair."

### Slide 5 — PSO: the method and why
The three forces; the velocity and position equations; $w = 0.7298$, $c_1 = c_2 = 1.49618$ (Clerc–Kennedy
constriction, φ = 4.1); $v_{\max} = 0.2 \times$ range; absorbing wall; decode = `clip(rint(x))`.

*Say:* "Three forces: inertia is momentum, the cognitive term pulls to the particle's own best, the social term
pulls to the swarm's best. The coefficients are the Clerc–Kennedy constriction values, which guarantee the swarm
is stable rather than divergent — so I did not have to tune the tuner. The state stays **continuous** and I
round only at evaluation time; if I rounded the state, any velocity below 0.5 would be annihilated and particles
would freeze."

### Slide 6 — Program snippet
The `velocity_update` + `apply_absorbing_bounds` functions, and the `make_objective` adapter showing
`fitness=...` versus `info={...}`.

*Say:* "The update rules are pure functions of arrays, so they are unit-tested against hand-computed values. And
this adapter is the sensor specification: **only the scalar fitness goes back to PSO.** Balanced accuracy and
macro F1 are computed every evaluation and written to disk, but they are structurally incapable of influencing
the search."

### Slide 7 — No leakage (the credibility slide)
Nested 5×5 CV diagram; `HeldOutTestSet.reveal()` raising inside `OptimizationPhase`; the IT-06/IT-07 label and
feature invariance tests; the log excerpt showing phase-closed before reveal.

*Say:* "Test isolation here is not a convention, it is enforced at runtime. Test arrays are private and reachable
only through `reveal()`, which **raises an exception** if an optimization phase is open. And I prove it: I
scramble every label in the test fold, re-run the whole optimization, and the trajectory is bit-for-bit
identical. If any test data had leaked, those runs would diverge."

### Slide 8 — Results
The §9.1 test-accuracy table **next to** the §9.2 validation-fitness table, plus 4 W / 6 T / 5 L, plus figure F4
(the anytime curve).

*Say:* "PSO beats the default RF on validation fitness on all three datasets — the optimizer does its job. But
on held-out test accuracy it is 4 wins, 6 ties, 5 losses. That is a tie, and I report it as a tie. Against
equal-budget random search it is 4–8–3, also inside noise. With five paired folds the minimum Wilcoxon p-value
is 0.0625, so significance is arithmetically impossible — I give means, spreads and win/tie/loss, and no
p-values."

### Slide 9 — Why — and what it teaches
Three per-dataset reasons (saturation / depth bound / noise); the winner's curse (Heart: 0.8415 validation vs
0.8283 test); the flat-landscape explanation for why random search competes; the 10–24 % cache-hit evidence of
convergence.

*Say:* "Each dataset fails to benefit for a different reason. Iris is saturated. Digits is limited by my own
`max_depth ≤ 20` bound — the default grows unlimited-depth trees, and I flagged that risk in an ADR before
running. Heart has the most validation headroom, +2.9 points, but only 61 test samples, so the gain cannot be
resolved from noise. And the general lesson: **feedback buys little when the landscape is flat.** PSO revisits
10 to 24 % of its evaluations as the swarm contracts, so it reaches the same ceiling with fewer distinct model
trainings — but on a flat, three-dimensional box, random sampling is a genuinely strong competitor. That tells
you *when* closed-loop optimization is worth the complexity."

### Slide 10 — Validation and conclusion
203/203 tests passing; `verify`: OK; `git_dirty: false`; config hash; 62.7 min runtime; deployment
configurations.

*Say:* "203 tests pass, including eight dedicated isolation and closed-loop proofs. An independent verifier
re-reads every result file, recomputes the summaries from the raw per-run artifacts, and checks the log ordering
— it reports OK. The experiment ran from a clean committed tree, so the exact source that produced these
numbers is recoverable from the commit hash. The conclusion: the closed loop is real, provable and correctly
implemented; its benefit on these three datasets is a measured null result; and I can explain exactly why."

---

# PART 13 — VIVA QUESTIONS AND ANSWERS

## On PSO fundamentals

**Q: Why PSO and not a Genetic Algorithm?**
A: Both are population-based and derivative-free. PSO suits this problem better for three reasons: the search
space is a **bounded continuous box**, which PSO handles natively while a GA would need crossover and mutation
operators designed for mixed integer ranges; PSO has **fewer parameters** to justify; and PSO's particles retain
**individual memory** (pbest), so information is shared without discarding individuals the way selection does.
With only 210 evaluations, I cannot afford a GA's generational turnover.

**Q: Where do 0.7298 and 1.49618 come from?**
A: They are the **Clerc–Kennedy constriction coefficients**. With $\varphi = 4.1$, the constriction factor
$\chi = 2/|2 - \varphi - \sqrt{\varphi^2 - 4\varphi}| \approx 0.72984$, giving $w = \chi$ and
$c_1 = c_2 = \chi \cdot \varphi/2 \approx 1.49618$. The point of the construction is a **stability guarantee**:
the swarm converges rather than oscillating with growing amplitude. They are the literature-standard canonical
values, which also means I did not tune PSO's own parameters — doing so would have made the comparison against
random search unfair.

**Q: What happens if you increase the inertia weight?**
A: More exploration, slower convergence. Above roughly 1.0 the velocity magnitude grows without bound and the
swarm diverges. Too low — say 0.3 — and the swarm collapses onto gbest in a few iterations and stops searching.
0.7298 is the stability-derived balance.

**Q: Why is $c_1 = c_2$?**
A: Equal weight to individual experience and social information, which is the canonical setting. Raising $c_2$
makes the swarm converge faster but risk premature convergence onto an early gbest; raising $c_1$ makes
particles more independent and slower to agree.

**Q: Why clamp the velocity?**
A: Without a clamp, a particle far from gbest gets a huge pull and can cross the whole box in one step, which
degenerates the search into random sampling. Clamping at 0.2 × range means no particle moves more than 20 % of a
dimension's span per iteration, preserving the local trajectory character that makes PSO a search rather than a
sampler.

**Q: Why an absorbing boundary and not a reflecting one?**
A: Because **the optimum may legitimately be on the boundary**. Accuracy is roughly non-decreasing in
`n_estimators`, so the best value is plausibly 200. Absorbing lets a particle sit *on* the bound; reflecting
would bounce it away from the point I want. This was confirmed: on Heart fold 0, PSO selected
`n_estimators = 200` exactly, and the run records a boundary hit.

**Q: Why strict `>` for pbest instead of `>=`?**
A: Three reasons. The fitness surface is full of exact ties, so with `>=` the best position would jitter between
equal-fitness points and the convergence curve would be meaningless. Strict `>` makes gbest fitness **provably
non-decreasing**, which is what makes the convergence plot valid. And it makes tie-breaking deterministic, so a
given seed reproduces a given trajectory.

**Q: Why gbest and not a ring topology?**
A: With 10 particles and 20 iterations, I cannot afford slow information propagation. In a ring, news of a good
configuration takes several iterations to cross the swarm, and I only have 20. gbest spreads it immediately.
The trade-off is a higher risk of premature convergence, which the diversity curve (F11) lets me monitor.

**Q: How do you handle integer hyperparameters?**
A: The swarm state stays **continuous**; I decode with `clip(rint(x))` only at evaluation, and never write the
rounded value back. If I rounded the state, a particle with a velocity component below 0.5 would round back to
the same integer every iteration and be **frozen forever** in that dimension. Keeping the state continuous lets
small velocity contributions accumulate: 8.0 → 8.3 → 8.6, and at 8.6 the decoded value finally flips to 9.

**Q: Did PSO get stuck in a local optimum?**
A: On Iris, the best configuration was often found at iteration 0 and never improved, which looks like getting
stuck but is actually the surface being **flat at the top** — many configurations tie for best. The evidence is
RQ4: across folds, `max_depth` ranged from 2 to 14 while test accuracy was identical. The problem is
underdetermined, not the optimizer trapped.

## On the closed loop

**Q: Is this really closed-loop control?**
A: It is a genuine closed **optimization** loop, and I am precise about what that does and does not mean. What
makes it closed: the controller's output changes a system parameter, the system responds, the response is
measured, and the next output is a function of that measurement — formally,
$x^{t+1} = \Phi(x^t, v^t, p^t, g^t, r^t)$ where $p^t$ and $g^t$ are functions of the feedback history. What it
is **not**: the plant is static, with no dynamics or time constant; there is no setpoint, so this is
extremum-seeking rather than reference tracking; and the controller acts through memory rather than an error
signal. I prove the loop formally, and I prove it empirically with a feedback-ablation test that shows removing
the feedback degrades the search.

**Q: How do I know the feedback actually matters?**
A: Integration test **IT-02** is a feedback ablation: it zeroes the cognitive and social contributions and
confirms the search becomes measurably less effective. And the whole experiment is designed as the controlled
version of that question — random search has the same budget, same objective, same folds, same seed, and
differs *only* in that it ignores feedback until the end.

**Q: Why is random search the right control?**
A: Because it isolates the single variable of interest. Same budget of 210, same objective $F_k$, same inner
folds, same seed, and a fresh cache each so neither free-rides. Everything is held constant **except the
presence of feedback**. Any systematic difference is therefore attributable to the feedback itself.

**Q: Where is the loop actually wired in the code?**
A: In exactly one place: `src/pso_rf/experiments/runner.py`. That is an enforced architectural rule — the
optimizer package imports only NumPy and the standard library, the evaluation package never imports the
optimizer, and unit test UT-22 walks the transitive imports to prove it. There is no disconnected PSO script
anywhere in the repository.

## On methodology and leakage

**Q: How do you know there is no test-set leakage?**
A: Four structural mechanisms and eight tests. The test arrays are private and reachable only through
`reveal()`, which **raises `TestSetAccessError`** if an optimization phase is open; every optimization is wrapped
in that phase; the optimizer receives no data at all, only a search space and a callable; and the fitness
evaluator rejects a test set if one is ever handed to it. The proofs: IT-06 scrambles every test **label**,
IT-07 scrambles every test **feature**, and both assert the PSO trajectory is bit-for-bit identical. IT-04
checks zero index overlap. ET-04 checks the timestamps. And `verify` parses the whole log to confirm every
reveal came after a phase close.

**Q: Why nested cross-validation instead of a single train/test split?**
A: A single split gives one noisy estimate and wastes data. Nested CV means **every row serves as test data
exactly once**, so I get five paired comparisons per dataset and a legitimate pooled confusion matrix — while
keeping hyperparameter selection strictly inside the optimization portion of each fold.

**Q: Why 5 inner folds when your proposal said 3?**
A: Recorded as ADR-005. Fitness resolution is $1/n$ regardless of *k*, so *k* does not change precision — it
changes the **training-fold size**: 67 % for 3-fold versus 80 % for 5-fold. Since `max_depth` and
`min_samples_split` act on absolute sample counts, tuning at 80 % transfers better to the final refit at 100 %.
It also lowers estimator variance, and because the folds run in parallel the measured extra cost was only
4–20 %. I built a timing gate into the plan: if the projection had exceeded 2 hours I would have fallen back to
3-fold and recorded it. The projection was 37.5 minutes, so 5-fold stayed.

**Q: Why are the inner folds frozen for the whole run?**
A: Common random numbers. If I re-drew folds per evaluation, the fitness would become stochastic, the cache
would be invalid, and — worst — **pbest would reward luck**: a mediocre configuration that drew an easy split
would be recorded as a personal best and would pull its particle, and through gbest the whole swarm, in the
wrong direction.

**Q: Why no p-values?**
A: With 5 paired folds, the minimum two-sided Wilcoxon signed-rank p-value is **0.0625** — significance at 0.05
is arithmetically impossible. And CV folds are not independent, since they share training data, so the test's
assumptions fail anyway. I report means, standard deviations, paired per-fold deltas, win/tie/loss counts and
pooled metrics, and I describe differences of one or two test samples as ties.

**Q: Your baseline has `max_depth=None`, outside your search space. Isn't that unfair?**
A: It is deliberately the honest comparison — it is what a practitioner gets by typing
`RandomForestClassifier()`. It does mean a PSO loss may be caused by the bound rather than the optimizer, and I
flagged exactly that risk in ADR-003 **before** running. The Digits results are consistent with it: PSO pushed
`max_depth` to 19 against the cap of 20 on fold 0. So I can separate "the optimizer underperformed" from "the
search space was too small", and on Digits it is partly the latter.

**Q: How did you handle the missing values in Heart Disease?**
A: Four missing in `ca`, two in `thal`, encoded as `?` in the raw file. I impute with the most frequent value —
but critically, the imputer is **the first step of a scikit-learn Pipeline**, so `cross_validate` refits it
inside each fold on that fold's training rows only. Imputing on the whole dataset before splitting would leak
the mode from validation and test rows into training. Unit test UT-20 asserts the imputer's statistics come from
training rows only.

## On the results

**Q: Your PSO didn't beat the baseline. Isn't the project a failure?**
A: No — and I would argue the null result is more informative than a win. Three things to separate. First, the
optimizer **works**: PSO's validation fitness beat the default RF on all three datasets, by up to 2.9 points on
Heart. It maximized the objective it was given. Second, that gain did not transfer to held-out test accuracy,
which is a textbook demonstration of the **generalization gap and the winner's curse** — the maximum of 210
noisy estimates is biased upward, and I measured the bias at 1.3 points on Heart. Third, I can name *why* each
dataset failed to benefit: Iris is saturated, Digits is limited by my own depth bound, and Heart has only 61
test samples. A lucky win on one dataset would have taught me less than this does.

**Q: Why does random search do just as well?**
A: Because the fitness landscape is **flat near the optimum**. Look at the validation fitness: Digits 0.9755139
for PSO versus 0.9755120 for random search — both are hitting the same ceiling. When many configurations tie at
the top, you do not need a clever trajectory; you need to sample anywhere in a large good region, which random
search does well. Feedback buys you something when the landscape is rugged or high-dimensional. On a flat,
three-dimensional box it buys you efficiency — PSO reached the same ceiling using 10 to 24 % fewer distinct
model trainings — but not a better endpoint. Knowing *when* closed-loop optimization pays off is a real finding.

**Q: Your chosen hyperparameters vary wildly across folds. Isn't that instability?**
A: It is instability in the *selected configuration*, not in the *performance*. On Iris, `max_depth` ranged from
2 to 14 across folds while test accuracy was **identical on all five folds**. That tells me the optimum is a
large flat region containing many equally good configurations — the problem is underdetermined. It also
disconfirmed one of my own prior expectations: ADR-003 predicted `n_estimators` would drift to the upper bound of
200, and only 1 of 15 folds actually hit it, which means accuracy had already saturated well below 200 trees.

**Q: What would you do with more time?**
A: Four things, in priority order. **Add `max_features` to the search space** — Section 2.1's variance
decomposition shows it controls the inter-tree correlation that sets the floor of ensemble variance, so it is
likely the highest-leverage knob and I am not tuning it. **Raise the `max_depth` bound** for Digits, since the
default's unlimited depth beat my cap of 20. **Repeat the outer CV with several seeds** to separate optimizer
variance from fold variance. And **test on a harder dataset with real headroom** — on sets already at 96–97 %
accuracy, no optimizer can demonstrate much.

**Q: Why did your experiment take 62.7 minutes when you projected 37.5?**
A: The benchmark used short, warm, repeated calls and underestimated **sustained** cost by about 2.7×. The likely
causes are per-call parallel dispatch overhead and sustained-load CPU clocking, but I have not proven that, and
the ADR says so explicitly rather than asserting a cause I cannot demonstrate. It still met the "about one hour"
requirement, and the 2-hour fallback threshold that would have forced 3-fold CV was never reached.

## On the engineering

**Q: Did you use a PSO library?**
A: No — libraries are forbidden by the brief, and PSO is implemented from scratch in NumPy. pyswarms, Optuna,
scikit-optimize, Hyperopt, `GridSearchCV` and `RandomizedSearchCV` are all prohibited. The optimizer package
imports **only NumPy and the standard library**, and unit test UT-22 enforces that by walking the transitive
import graph.

**Q: How do I know your reported numbers match your raw data?**
A: `python -m pso_rf verify` **recomputes** `summary.csv` and `summary_folds.csv` from the raw per-run
`final.json` files and requires an exact match. You cannot hand-edit a summary and pass. It also checks that
every evaluation CSV has the right row count, that pooled predictions cover every sample exactly once, and that
every test reveal in the log came after a phase close. It reports `OK: all checks passed`.

**Q: Can you reproduce the experiment exactly?**
A: Yes. One NumPy `Generator` per optimizer and no global random state; a documented fixed RNG draw order
(positions, velocities, then $R_1$/$R_2$ per iteration); all seeds in config; a SHA-256 config hash; dataset
checksums including the raw Heart file; pinned dependency versions; and the git commit with
`"git_dirty": false`, so the exact source is recoverable. Test IT-08 re-runs a configuration and asserts every
result file is byte-identical apart from timing fields.

**Q: What are those two `NotImplementedError`s in your code?**
A: Both are deliberate fail-loud guards, not missing features. The one in `pso.py` refuses any boundary,
topology or update mode other than the absorb/gbest/synchronous variant the design specifies — reflecting
boundaries and ring topology are documented extensions I never validated, so the code will not silently run an
untested path. The one in `audit.py` refuses to proceed if duplicate rows exceed 1 % of a dataset, because the
requirements say such a dataset needs deduplication before splitting; measured duplicates are 0.67 % on Iris and
zero elsewhere, so it never fires. It exists so a future dataset cannot slip past the rule unnoticed.

---

# PART 14 — NUMBERS TO MEMORIZE

Learn this table. It covers almost every number you will be asked for.

## Search space and PSO

| Quantity | Value |
|---|---|
| Decision variables | 3: `n_estimators`, `max_depth`, `min_samples_split` |
| Bounds | (50–200), (2–20), (2–10) |
| Ranges | (150, 18, 8) |
| **Space size** | **25,821** = 151 × 19 × 9 |
| Particles *N* | **10** |
| Iterations *T* | **20** |
| **Evaluations per run** | **210** = 10 × 21 |
| Inertia *w* | **0.7298** |
| $c_1 = c_2$ | **1.49618** |
| $v_{\max}$ = 0.2 × range | **(30, 3.6, 1.6)** |
| $v_{\text{init}}$ = 0.1 × range | (15, 1.8, 0.8) |
| Topology / update / boundary | gbest / synchronous / absorbing |
| Patience | available, **disabled** (tol 1e-4, P = 5) |

## Protocol

| Quantity | Value |
|---|---|
| Outer folds / seed | 5 / **42** |
| Inner folds / seed | 5 / run seed *k* |
| Run seeds | 0, 1, 2, 3, 4; deployment 5 |
| Methods | baseline, random_search, pso |
| Runs | **48** = 3 × 5 × 3 + 3 deployment |
| Total fitness evaluations | **6,945** |
| Baseline config | (100, None, 2) — **outside** the search space |
| Class-ratio gate | 1.5 — **never triggered** (max ratio 1.18) |

## Datasets

| | Iris | Digits | Heart |
|---|---|---|---|
| Samples | 150 | 1,797 | 303 |
| Features | 4 | 64 | 13 |
| Classes | 3 | 10 | 2 |
| Class ratio | 1.000 | 1.052 | 1.180 |
| Test fold | 30 | ~359 | 61 |
| Opt portion | 120 | 1,437 | 242 |
| Missing | 0 | 0 | 6 (`ca` 4, `thal` 2) |
| Duplicates | 1 (kept) | 0 | 0 |

## Results — the two tables that matter

**Held-out test accuracy (mean ± sd):**

| | Baseline | Random search | PSO |
|---|---|---|---|
| Iris | 0.9600 ± 0.0435 | 0.9533 ± 0.0380 | **0.9600 ± 0.0435** |
| Digits | 0.9722 ± 0.0081 | 0.9750 ± 0.0028 | **0.9744 ± 0.0046** |
| Heart | 0.8316 ± 0.0359 | 0.8251 ± 0.0412 | **0.8283 ± 0.0478** |

**Validation fitness (what PSO optimizes):**

| | Baseline | Random search | PSO |
|---|---|---|---|
| Iris | 0.9483 | 0.9583 | **0.9583** |
| Digits | 0.9713 | 0.9755120 | **0.9755139** |
| Heart | 0.8126 | 0.8414286 | **0.8414626** |

**Win / tie / loss over 15 folds:** PSO vs baseline **4 / 6 / 5** · PSO vs random search **4 / 8 / 3**

**Cache repeat rate (unique fits out of 210):** Iris 188.8 · Digits 159.0 · Heart 164.2 (RS ≈ 209.6 for all)

**Deployment configs:** Iris (171, 17, 6) · Digits (178, 18, 3) · Heart (110, 3, 4)

**Best single result:** Heart fold 0, PSO 0.9016 vs baseline 0.8689 → **+3.28 points**
**Worst single result:** Heart fold 2, PSO 0.8033 vs baseline 0.8525 → **−4.92 points**

## Engineering

| Quantity | Value |
|---|---|
| Tests | **203 passing** (3 min 22 s) |
| Source files / lines | 47 / ~5,500 |
| Test files / lines | 27 / ~2,720 |
| Docs / ADRs | 14 documents / **27 ADRs** |
| Figures | 9 (F3–F11) |
| **Experiment runtime** | **62.7 min** |
| Median seconds per evaluation | Iris 0.354 · Digits 1.000 · Heart 0.350 |
| Config hash | `80474a0c3a1485fe...` |
| Git commit | `8bd5927`, `git_dirty: false` |
| Python / sklearn | 3.10.11 / 1.7.2 |

## Three sentences that cover 80 % of the viva

1. **"PSO is the controller, RF training plus inner-CV is the plant, inner-CV accuracy is the feedback, and the
   next proposal is a function of that feedback — which is what makes the loop closed."**
2. **"The test fold is read exactly once, after the loop terminates; `reveal()` raises an exception inside an
   optimization phase, and I prove independence by scrambling every test label and showing the trajectory is
   bit-for-bit identical."**
3. **"PSO maximized its objective on all three datasets but that did not transfer to test accuracy — 4 wins, 6
   ties, 5 losses — because the landscape is flat, the datasets are near their ceiling, and the validation
   maximum is optimistically biased. I report that as a null result."**

---

# PART 15 — COMMANDS, FILE MAP, GLOSSARY

## 15.1 Commands

```bash
# Install (editable)
pip install -e .

# Full experiment (~63 min)
python -m pso_rf run --config configs/default.yaml

# Quick demo config (Iris, fold 0, 6 particles × 5 iterations)
python -m pso_rf run --config configs/demo.yaml

# Verify saved results (completeness, coverage, summary recomputation, log ordering)
python -m pso_rf verify --results results/20260925-083324_default

# Regenerate figures F3–F11
python -m pso_rf plot --results results/20260925-083324_default

# Compare two result directories (ignoring timing)
python -m pso_rf compare --a results/<A> --b results/<B>

# Dataset audit
python -m pso_rf audit

# Interactive frontends (Review 2 demo)
python -m pso_rf web          # Tornado "Swarm Lab" — 3-D live swarm
python -m pso_rf demo         # Streamlit demo
streamlit run app.py          # same Streamlit app directly

# Tests
python -m pytest tests/ -q                      # all 203
python -m pytest tests/ -m isolation -q         # isolation proofs only
python -m pytest tests/ -m closed_loop -q       # closed-loop proofs only
```

## 15.2 Repository map

```
PSO/
├── CLAUDE.md                  master development rules
├── README.md                  project overview and install
├── project.md                 ← THIS HANDBOOK
├── app.py                     Streamlit entry shim
├── pyproject.toml             package, pinned deps, pytest + ruff config
├── requirements.txt           pinned dependencies
│
├── configs/
│   ├── default.yaml           the production experiment
│   ├── demo.yaml              Iris, fold 0, tiny budget
│   └── test.yaml              all datasets, 4 particles × 2 iterations
│
├── data/
│   ├── raw/processed.cleveland.data   vendored, SHA-256 in MANIFEST.json
│   └── DATASET_AUDIT.md
│
├── docs/   (13 documents)
│   ├── DECISIONS.md                   27 ADRs ← the "why" of everything
│   ├── MATHEMATICAL_FORMULATION.md    all equations
│   ├── ARCHITECTURE.md                all diagrams
│   ├── CONTEXT.md                     interfaces, confirmed/proposed/to-verify
│   ├── REQUIREMENTS.md                numbered requirements + traceability
│   ├── EXPERIMENT_PLAN.md             RQ1–RQ5, run matrix, planned plots
│   ├── TESTING_STRATEGY.md            every test ID
│   ├── RESULTS_SCHEMA.md              exact schema of every output file
│   ├── IMPLEMENTATION_PLAN.md         phases P0–P15
│   ├── PROJECT_STRUCTURE.md           layout + layering rule
│   ├── IDEA.md                        original proposal
│   ├── REVIEW_1_MAPPING.md            Review 1 evidence
│   └── REVIEW_2_MAPPING.md            Review 2 evidence
│
├── src/pso_rf/
│   ├── optimization/   pso, random_search, search_space, events   ← NumPy only
│   ├── evaluation/     fitness, splits, final, metrics            ← the plant + guard
│   ├── datasets/       iris, digits, heart_cleveland, audit, base
│   ├── preprocessing/  pipeline
│   ├── models/         random_forest, baseline
│   ├── experiments/    runner ← THE LOOP, config, recorder, summary, verify, cli, seeding
│   ├── visualization/  plots (F3–F11)
│   ├── utils/          io, hashing, log, manifest
│   ├── app/            Streamlit demo (ADR-026)
│   └── web/            Tornado "Swarm Lab" (ADR-027)
│
├── tests/
│   ├── unit/           14 files — includes UT-15, UT-20, UT-22
│   ├── integration/    6 files  — IT-01…IT-07
│   ├── experiment/     1 file   — ET-01…ET-04, IT-08
│   └── app/            2 files
│
├── results/20260925-083324_default/        ← THE EXPERIMENT OF RECORD
│   ├── manifest.json            git SHA, versions, checksums, timings
│   ├── config.resolved.json     fully resolved config
│   ├── summary.csv              per dataset × method
│   ├── summary_folds.csv        per dataset × method × fold
│   ├── run.log                  full trace incl. phase transitions
│   └── <dataset>/
│       ├── audit.json
│       ├── fold_<k>/<method>/{final.json, predictions.csv, evaluations.csv, iterations.csv}
│       └── deployment/pso/{deployment.json, evaluations.csv, iterations.csv}
│
├── plots/<exp_id>/    F3…F11 PNGs + SOURCES.json
└── scripts/           download_cleveland, audit_datasets, benchmark_eval,
                       measure_landscape, report_tables
```

## 15.3 Output file schemas (what to point at if asked for evidence)

| File | One row / record per | Key fields |
|---|---|---|
| `evaluations.csv` | **every single evaluation** | dataset, method, fold, seed, iteration, particle, position, velocity, hyperparameters, fitness, pbest, gbest, cv_scores, diagnostics, cache_hit, fit_time_s |
| `iterations.csv` | every PSO iteration | gbest fitness and config, gbest_improved, mean/std/min/max fitness, **diversity**, n_unique_configs, n_ties_with_gbest, n_failed, cumulative_evaluations, elapsed_s |
| `final.json` | one run | best_hyperparameters, best_validation_fitness, validation_cv_scores, best_found_at_eval/iteration, convergence_iteration, convergence_status, n_ties_with_best, **boundary_hits**, test_metrics (incl. confusion matrix), optimization_finished_at, test_evaluated_at |
| `predictions.csv` | one test row | row_index, y_true, y_pred |
| `deployment.json` | one deployment run | recommended_hyperparameters, validation_fitness, performance_estimate, the honest `note` |
| `summary_folds.csv` | dataset × method × fold | chosen hyperparameters, validation fitness, all test metrics, **paired deltas**, n_unique_fits, convergence_iteration |
| `summary.csv` | dataset × method | mean ± sd of test metrics, pooled accuracy, mean validation fitness, delta vs baseline, **W/T/L vs baseline and vs RS**, mean unique fits, total time |
| `manifest.json` | one experiment | exp_id, timestamps, command, config_hash, **git_commit + git_dirty**, Python/platform/CPU, package versions, dataset checksums, status |

## 15.4 Glossary

| Term | Meaning |
|---|---|
| **Particle** | One candidate solution — a point in the hyperparameter space |
| **Position** $x$ | The particle's current (continuous) candidate |
| **Velocity** $v$ | The step to be added to position next iteration |
| **pbest** $p$ | Best position this particle has ever visited (cognitive memory) |
| **gbest** $g$ | Best position any particle has visited (social memory) |
| **Inertia** $w$ | Fraction of previous velocity retained |
| **Cognitive coefficient** $c_1$ | Strength of the pull towards pbest |
| **Social coefficient** $c_2$ | Strength of the pull towards gbest |
| **Constriction coefficient** | Clerc–Kennedy factor $\chi$ guaranteeing swarm stability |
| **Absorbing boundary** | Clamp position to the wall and zero that velocity component |
| **Decode** | `clip(rint(x), l, u)` — continuous position → integer configuration |
| **Diversity** | Mean normalized distance of positions from the swarm centroid |
| **Synchronous update** | Evaluate all particles, then update memories, then move |
| **Fitness** | Mean inner-CV accuracy $F_k(\theta)$ — the feedback signal |
| **Convergence iteration** $t_c$ | Last iteration at which gbest improved |
| **Plant** | The system under control: RF training + inner-CV validation |
| **Open loop** | Random search — proposals ignore feedback until the end |
| **Nested CV** | Outer folds for honest testing, inner folds for fitness |
| **Stratified** | Each fold preserves the class proportions of the whole set |
| **Winner's curse** | Upward bias in the maximum of many noisy estimates |
| **Optimization portion** | $\mathcal{D} \setminus P_k$ — everything PSO is allowed to see |
| **Held-out test fold** | $P_k$ — sealed until the loop terminates; read once |
| **Deployment run** | PSO on all data (seed 5); has no test score by construction |
| **Bagging** | Bootstrap aggregating — each tree trained on a resample |
| **Out-of-bag** | The ~36.8 % of rows not in a given tree's bootstrap sample |
| **ADR** | Architecture Decision Record — a numbered, justified design decision |

## 15.5 ADR quick index

| ADR | Topic | ADR | Topic |
|---|---|---|---|
| 001 | PSO as the optimizer | 015 | Stopping and convergence |
| 002 | Random Forest as the plant | 016 | Fitness caching |
| 003 | The 3 hyperparameters and bounds | 017 | Tie-breaking |
| 004 | Validation fitness; test never used | 018 | Seed policy |
| 005 | Inner 5-fold CV, folds fixed | 019 | Parallelism |
| 006 | Accuracy as fitness (configurable) | 020 | Reporting rules (no p-values) |
| 007 | Outer 5-fold nested CV + deployment | 021 | Multi-dataset evaluation |
| 008 | Baseline + equal-budget random search | **022** | **Why the system is closed-loop** |
| 009 | Heart = UCI Cleveland, checksummed | 023 | Package name and import layering |
| 010 | Preprocessing inside the pipeline | **024** | **Test-set access guard** |
| **011** | **Continuous state, decode at evaluation** | 025 | Configuration format and layering |
| **012** | **Boundary handling and velocity limits** | 026 | Interactive Streamlit demo |
| 013 | PSO coefficients | 027 | Web frontend "Swarm Lab" |
| 014 | Swarm size, iterations, topology, update | | |

The five in bold are the ones most likely to come up. Read them in full in `docs/DECISIONS.md` the night before.

---

## FINAL CHECKLIST — NIGHT BEFORE

- [ ] Re-read **Part 0** (the 60-second answer) until you can say it without looking
- [ ] Memorize **Part 14** — especially the two results tables and 4/6/5 and 4/8/3
- [ ] Be able to **draw the block diagram from §6.1 from memory**, feedback arrow included
- [ ] Be able to **write the velocity equation from memory** and name all three terms
- [ ] Know **why 0.7298 / 1.49618** (Clerc–Kennedy constriction, φ = 4.1, stability)
- [ ] Know **why the state stays continuous** (particles would freeze if you rounded it)
- [ ] Know **why an absorbing wall** (the optimum may be on the boundary — and Heart fold 0 proves it)
- [ ] Know **why strict `>`** (plateau stability, monotone gbest, determinism)
- [ ] Be able to explain the **null result confidently**, with the three per-dataset reasons
- [ ] Be able to state the **honest limits** of the control framing *before* being challenged
- [ ] Read **ADR-011, 012, 022, 024** in full in `docs/DECISIONS.md`
- [ ] Have the terminal ready: `python -m pso_rf verify --results results/20260925-083324_default` → `OK`
- [ ] Have `python -m pso_rf web` ready for the live demo, and know that `app/explain.py` runs the isolation
      proof live in the browser
- [ ] Open figure **F4** (the anytime curve) — it is the best single visual for the closed-loop argument

---

*Handbook compiled 2026-10-06. Every quantitative claim traced to a file in
`results/20260925-083324_default/`. Test suite: 203 passing. Verification: `OK: all checks passed`.*
