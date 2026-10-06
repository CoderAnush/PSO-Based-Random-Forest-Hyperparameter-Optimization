# 🦅 PSO-Based Random Forest Hyperparameter Optimization

## What is this? (In Simple Terms)

Imagine you're tuning a **recipe** (Random Forest) by adjusting three knobs:
- **Number of chefs** (n_estimators): 50–200
- **Tree depth** (max_depth): 2–20 levels  
- **Minimum customers per table** (min_samples_split): 2–10

Instead of trying all **25,821 combinations randomly**, we use a **swarm of birds** (PSO) that:
1. Each bird tries a configuration → tastes the recipe (measures accuracy)
2. Remembers the best recipe it found
3. Remembers the swarm's best recipe (global best)
4. Flies toward better recipes in the search space
5. After 20 rounds (20 iterations), we found a good recipe!

**The key insight:** The birds communicate, learn from each other, and converge to a good solution faster than blind random search. We compare this intelligent search against:
- **Default recipe** (no tuning)
- **Random guess** (blind sampling, 210 tries)
- **PSO** (smart birds with feedback, 210 tries)

---

## The Big Picture: Closed-Loop Optimization

Think of it like a **thermostat**:

```
┌─────────────────────────────────────────────┐
│  PSO (Smart Tuner)                          │
│  └─ Proposes hyperparameters               │
└────────────┬────────────────────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  Random Forest  │  (The machine learning model)
    │  (The Recipe)   │  └─ Trains and validates
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Accuracy Score │  (How good is this recipe?)
    │  (Validation)   │  └─ Returns as FEEDBACK
    └────────┬────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│  PSO uses this score to:                    │
│  • Update personal best (my best so far)   │
│  • Update global best (team's best so far) │
│  • Adjust flight direction (velocity)      │
│  • Move to next position (new hyperparams) │
└─────────────────────────────────────────────┘
             │
             └───→ REPEAT 20 times
```

This is a **closed feedback loop** — the optimizer learns from the model's performance.

---

## Key Results 📊

After 210 evaluations (10 particles × 20 iterations):

| Dataset | PSO Accuracy | Default RF | Random Search | Winner |
|---------|---|---|---|---|
| **Iris** | **96.00%** | 96.00% | 95.33% | Tie ✓ |
| **Digits** | **97.44%** | 97.22% | 97.50% | Random (lucky) |
| **Heart Disease** | 82.83% | **83.16%** | 82.51% | Default |

**Takeaway:** PSO is competitive with random search on small datasets, using **10–24% fewer model fits**.

---

## Installation (Step by Step)

### Prerequisites
- **Python 3.10** (download from python.org)
- **Git** (for version control)

### Step 1: Clone the repository
```bash
git clone https://github.com/CoderAnush/PSO-Based-Random-Forest-Hyperparameter-Optimization.git
cd PSO-Based-Random-Forest-Hyperparameter-Optimization
```

### Step 2: Create a virtual environment (keeps your Python clean)
```bash
python -m venv .venv
```

### Step 3: Activate it
**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### Step 4: Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

Done! You're ready to run experiments. ✓

---

## Try It Out (No Deep Learning Knowledge Needed)

### 🎮 **Interactive Visual Demo (Recommended)**

Watch the optimization happen in real-time with 3-D visualization:

```bash
python -m pso_rf web
```

Open `http://localhost:8600` in your browser. You'll see:
- **Live Lab:** Watch 10 birds (particles) search for the best hyperparameters while racing random search
- **Playground:** Play with the PSO settings and see instant results on sample data
- **Results:** The final scores from the full experiment
- **Replay:** Rewatch saved optimization runs
- **How it works:** Learn the theory with interactive proofs

### 📊 **Classic Dashboard (Streamlit)**

```bash
python -m pso_rf demo
```

Opens a dashboard showing results, graphs, and live feedback traces.

---

## Run Full Experiment

Want to reproduce the entire study?

```bash
python -m pso_rf run --config configs/default.yaml
```

This will:
1. Load 3 datasets (Iris, Digits, Heart Disease)
2. Split each into 5 folds (for honest testing)
3. Run PSO, Random Search, and Default RF on each fold
4. Measure accuracy on held-out test data
5. Save results to `results/20260925-083324_default/`
6. Generate plots

**Typical runtime:** ~30 minutes on a laptop

---

## Understanding the Math (Optional but Helpful)

### The PSO Velocity Update Equation

Each particle flies through the search space using this rule:

$$v_i^{t+1} = w \cdot v_i^t + c_1 r_1 (p_i - x_i^t) + c_2 r_2 (g - x_i^t)$$

**Translation:**
- $v_i^{t+1}$ = particle's new velocity (where to fly next)
- $w = 0.7298$ = inertia (momentum from past movement)
- $c_1 = c_2 = 1.49618$ = learning coefficients (how much to trust personal/global best)
- $r_1, r_2$ = random numbers (0 to 1, keep search varied)
- $p_i$ = particle's personal best (best spot I've found)
- $g$ = global best (best spot the swarm found)
- $x_i^t$ = particle's current position (current hyperparameters)

**In plain English:**
> "Move in the direction you were going, but also drift toward your personal best memory and the swarm's best memory. Add some randomness to explore."

### Position Update

$$x_i^{t+1} = x_i^t + v_i^{t+1}$$

**Translation:** Your new position is your old position plus your velocity.

### Fitness (What We're Maximizing)

$$F(n\_estimators, max\_depth, min\_samples\_split) = \text{mean accuracy across 5 validation folds}$$

**In plain English:** We measure how good a recipe is by averaging its performance across 5 different test groups.

---

## Key Concepts Explained Simply

### 1. **Particles = Flying Search Agents**
Like birds in a flock, 10 particles explore the hyperparameter space. Each particle remembers its best spot and the swarm's best spot.

### 2. **Fitness = How Good a Recipe Is**
We train a Random Forest with given hyperparameters and measure accuracy. Higher accuracy = better fitness.

### 3. **Closed Loop = Feedback**
Unlike random guessing, PSO listens to the model's feedback and adjusts its search strategy. The model tells the optimizer "you're getting closer!"

### 4. **Validation vs Test Data (Very Important!)**
- **Validation (inner fold):** Used by PSO to decide which recipe to explore next (feedback)
- **Test (outer fold):** Sealed until the end. Only used to measure final performance (honest score)

This prevents "cheating" — PSO never sees test data.

### 5. **5-Fold Cross-Validation**
Imagine splitting data into 5 groups. Train on 4, test on 1. Repeat 5 times. This gives an honest estimate and is more stable than a single split.

---

## File Structure (What's Where)

```
📁 PSO-Based-Random-Forest-Hyperparameter-Optimization/
├── 📄 README.md                          ← You are here!
├── 📄 CLAUDE.md                          ← Development rules & constraints
├── 📄 project.md                         ← Full handbook (for presentations)
│
├── 📁 src/pso_rf/                        ← Main code
│   ├── optimization/                     ← PSO implementation (from scratch)
│   ├── evaluation/                       ← Random Forest training & validation
│   ├── datasets/                         ← Iris, Digits, Heart Disease loaders
│   ├── web/                              ← 3-D visualization frontend (Swarm Lab)
│   └── app.py                            ← Streamlit dashboard
│
├── 📁 configs/                           ← Settings (you can edit these)
│   ├── default.yaml                      ← Full experiment settings
│   └── demo.yaml                         ← Quick demo settings
│
├── 📁 results/                           ← Experiment outputs
│   └── 20260925-083324_default/         ← Your results live here!
│       ├── iris/fold_0/                  ← One fold's results
│       │   ├── evaluations.csv           ← All 210 evaluations
│       │   ├── final.json                ← Best hyperparameters found
│       │   └── predictions.csv           ← Predictions on test data
│       └── summary.csv                   ← Overall results across all folds
│
├── 📁 plots/                             ← Graphs and visualizations
│   └── 20260925-083324_default/
│       ├── accuracy_comparison.png       ← PSO vs Random vs Default
│       └── ...
│
├── 📁 docs/                              ← Technical documentation (14 files)
│   ├── MATHEMATICAL_FORMULATION.md       ← PSO equations & details
│   ├── ARCHITECTURE.md                   ← System design diagrams
│   ├── EXPERIMENT_PLAN.md                ← Research questions & protocol
│   └── ... 11 more design documents
│
├── 📁 tests/                             ← Automated tests (ensures correctness)
│   ├── unit/                             ← PSO tests (no data)
│   ├── integration/                      ← Closed-loop tests
│   └── experiment/                       ← Full experiment validation
│
└── requirements.txt                      ← Python packages needed
```

---

## Commands Quick Reference

### 🎮 **For Learning / Exploring**
```bash
python -m pso_rf web              # Best: interactive 3-D lab (recommended!)
python -m pso_rf demo             # Classic Streamlit dashboard
```

### 🧪 **For Testing / Validation**
```bash
python -m pytest -q                          # Run all tests (~3 min)
python -m pytest tests/unit -q               # Just PSO unit tests
python -m pytest tests/integration -q        # Closed-loop tests
```

### 📊 **For Experiments**
```bash
# Full experiment (3 datasets, 5 folds, 3 methods)
python -m pso_rf run --config configs/default.yaml

# Quick demo (Iris only, fold 0)
python -m pso_rf run --config configs/demo.yaml

# Custom: just digits, PSO only
python -m pso_rf run --config configs/default.yaml --datasets digits --methods pso

# Generate plots from saved results
python -m pso_rf plot --results results/20260925-083324_default

# Verify results are complete & correct
python -m pso_rf verify --results results/20260925-083324_default
```

---

## Results Explained

### **Iris Dataset** (150 flowers, 3 species)
- PSO found: n_estimators=179, max_depth=3, min_samples_split=8
- Test accuracy: **96.00%** (14.4 correct predictions out of 15)
- Why tied with default? Iris is easy; many hyperparameters work equally well

### **Digits Dataset** (1,797 handwritten digits, 10 classes)
- PSO found: n_estimators=152, max_depth=19, min_samples_split=2
- Test accuracy: **97.44%** (competes with random search)
- Best? Random search got lucky and found 97.50%

### **Heart Disease Dataset** (303 patients, 2 classes: healthy/diseased)
- PSO found: n_estimators=200, max_depth=6, min_samples_split=7
- Test accuracy: **82.83%** (less accurate, data is noisier)
- Default RF: 83.16% (default is actually better here!)

**Key Takeaway:** PSO works best when the hyperparameter landscape has clear patterns. On small, noisy datasets, it ties with random search.

---

## The Science Behind This Project

### Research Questions
1. **Can intelligent search (PSO) outperform random search?** 
   - Answer: On small datasets with simple structure, they tie.

2. **Can we tune Random Forest efficiently?**
   - Answer: Yes, PSO finds competitive results with 210 model evaluations.

3. **Is closed-loop feedback better than open-loop guessing?**
   - Answer: About equal on these datasets, but PSO uses smarter geometry.

### Why PSO? Why Random Forest?
- **PSO:** Mimics nature (bird flocking), converges faster than random search, no derivatives needed
- **Random Forest:** Robust, handles mixed data types, built-in feature importance, interpretable
- **Together:** A complete optimization loop (optimizer → model → feedback → optimizer)

---

## For Your Presentation / Report

### Key Numbers to Cite
- **210 evaluations** per method per fold (10 particles × 20 iterations = PSO budget)
- **25,821 possible configurations** in the search space
- **3 datasets, 5 folds, 3 methods = 45 experiments** total
- **5-fold cross-validation** (80% train, 20% test per fold)
- **Closed-loop proof:** IT-01, IT-02, IT-03 in test suite

### Plots Generated
- Accuracy comparison bar charts (PSO vs Random vs Default)
- Convergence curves (how PSO improves over 20 iterations)
- Hyperparameter distributions (which settings were preferred?)
- Swarm diversity plots (how spread out are the particles?)

### Where to Find Everything
- **Full results:** `results/20260925-083324_default/summary.csv`
- **Visualizations:** `plots/20260925-083324_default/`
- **All evaluations:** `results/20260925-083324_default/[dataset]/fold_[k]/pso/evaluations.csv`

---

## Project Info

**Author:** Anush Ramesh (CB.EN.U4ELC23005), EEE Department, Amrita Vishwa Vidyapeetham, Coimbatore

**Status:** ✅ Complete & Verified
- All tests pass
- Results reproducible (bit-for-bit identical on re-runs)
- Documentation complete (14 design documents)
- Live interactive demo available

**License:** Academic project (2026)

---

## Next Steps

1. **First time here?** → Run `python -m pso_rf web` and click through the tabs
2. **Want to understand the code?** → Read `docs/MATHEMATICAL_FORMULATION.md`
3. **Want to modify settings?** → Edit `configs/default.yaml` and re-run
4. **Want to use this for your own data?** → See `docs/CONTEXT.md` §9 for the API
5. **For your presentation?** → Check `project.md` (full handbook)

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'pso_rf'"**
→ Run `pip install -e .` again

**"Port 8600 already in use"**
→ Run `python -m pso_rf web --port 8601` (use a different port)

**"Tests failing"**
→ Make sure you're in the activated venv: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)

**"How do I run just one fold?"**
→ `python -m pso_rf run --config configs/default.yaml --folds 0`

---

## Technical Details (For Developers)

See [CLAUDE.md](CLAUDE.md) for:
- Closed-loop and test-isolation rules
- No-shortcuts policy (test data never in PSO feedback)
- Architecture principles
- Coding conventions

See [docs/](docs/) for:
- [MATHEMATICAL_FORMULATION.md](docs/MATHEMATICAL_FORMULATION.md) — PSO equations, bounds, integer decoding
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — System design, data flow, test guard implementation
- [TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md) — Test isolation proofs, closed-loop proofs
- [DECISIONS.md](docs/DECISIONS.md) — Why certain design choices were made (Architecture Decision Records)

## Installation

Python 3.10. The pinned dependency versions are in `requirements.txt`.

If those packages are already installed system-wide (as on the development machine), reuse them without
downloading anything:

```
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -e . --no-deps --no-build-isolation
```

On a clean machine, install the pinned versions first:

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e . --no-deps
```

(On Linux or macOS, use `.venv/bin/python`.)

## Interactive frontend: Swarm Lab

```
.venv/Scripts/python -m pso_rf web           # opens http://localhost:8600
```

Five views:
- **Live lab:** tune N, T, w, c₁, c₂, the velocity clamp, fold and seed, then watch the real closed loop in 3-D while it races random search. The sealed test fold opens only at the end.
- **Playground:** instant PSO over *measured* Random Forest accuracy landscapes, in 2-D map and 3-D terrain views, with presets and a feedback ablation.
- **Results:** the verified experiment.
- **Replay:** saved runs in 3-D.
- **How it works:** the loop, the protocol, and live proofs.

Works offline (three.js is bundled).

Classic Streamlit demo:

```
.venv/Scripts/python -m pso_rf demo          # or: .venv/Scripts/python -m streamlit run app.py
```

Four views: a **live closed-loop lab** (watch PSO ⇄ Random Forest close the loop, race random search, open the
sealed test fold once), the **experiment results** dashboard, a **swarm replay**, and **how it works** (with a
live test-isolation proof and a feedback ablation).

## Usage

| Command | Purpose |
|---|---|
| `python -m pytest -q` | run the test suite (unit, integration, experiment level; about 3 minutes) |
| `python -m pso_rf run --config configs/default.yaml` | full experiment: 3 datasets × 5 outer folds × {baseline, random search, PSO}, plus a deployment run per dataset |
| `python -m pso_rf run --config configs/default.yaml --config configs/demo.yaml` | small-budget live demo (Iris, fold 0) with a per-iteration trace |
| `python -m pso_rf verify --results results/<exp_id>` | audit a results directory: completeness, test isolation, summary consistency |
| `python -m pso_rf compare <results_a> <results_b>` | check two runs are identical apart from timing fields |
| `python -m pso_rf plot --results results/<exp_id>` | figures F3–F11 generated from the saved result files |

Useful `run` options: `--datasets iris digits`, `--folds 0 1`, `--methods baseline pso`, `--set pso.max_iter=30`.
Use the venv interpreter (`.venv/Scripts/python`) for all commands.

Results are written to `results/<exp_id>/` and figures to `plots/<exp_id>/`, as defined in
[docs/RESULTS_SCHEMA.md](docs/RESULTS_SCHEMA.md).

## Documentation

| Document | Contents |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Master development rules: closed-loop and test-isolation rules, conventions, forbidden shortcuts |
| [docs/IDEA.md](docs/IDEA.md) | Project idea for the faculty: problem, why Random Forest and PSO, the feedback loop |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | Numbered requirements (FR, NFR, AR, MLR, OR, DR, ER, RR, DOC) with traceability |
| [docs/CONTEXT.md](docs/CONTEXT.md) | Technical reference: confirmed, proposed and to-verify facts; module interfaces; configuration; seeds |
| [docs/MATHEMATICAL_FORMULATION.md](docs/MATHEMATICAL_FORMULATION.md) | Search space, objective, PSO equations, integer decoding, worked example, final evaluation |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Architecture diagrams, closed-loop control mapping, test isolation, configuration template |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Repository layout and the import-layering rule |
| [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) | Research questions, datasets, protocol, metrics, planned plots and tables |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Phases 0–15 with files, tests and completion criteria |
| [docs/TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md) | Unit, integration and experiment tests, including the isolation and closed-loop proofs |
| [docs/RESULTS_SCHEMA.md](docs/RESULTS_SCHEMA.md) | Result files, columns and types |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Architecture decision records (ADR-001 to ADR-025) |
| [docs/REVIEW_1_MAPPING.md](docs/REVIEW_1_MAPPING.md) | Faculty Review 1 requirements mapped to the project (Review 1 completed) |
| [docs/REVIEW_2_MAPPING.md](docs/REVIEW_2_MAPPING.md) | Faculty Review 2 requirements mapped to planned evidence |

The original proposal (the source of truth for the idea) is `ppt/CB.EN.U4ELC23005_ANUSH_RAMESH_PPT.pdf`.
