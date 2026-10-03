// Live closed-loop lab: the real PSO ⇄ Random Forest loop (experiment code path), streamed and drawn as it runs.
import { api, streamLive } from "../api.js";
import { LineChart, barChart } from "../charts.js";
import { LoopDiagram } from "../loop.js";
import { Swarm3D } from "../swarm3d.js";
import { COLORS, HP, LABEL, fmt, h, kpi, segmented, slider, toast, toggle } from "../ui.js";

const T_EVAL = { iris: 0.35, heart_cleveland: 0.35, digits: 1.0 }; // measured medians in the full experiment (s)
const STORE = "swarmlab.live.v1";

function saved() { try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch { return {}; } }
function save(v) { try { localStorage.setItem(STORE, JSON.stringify(v)); } catch { /* private mode */ } }

export function mount(root, meta) {
  const st = { dataset: "heart_cleveland", fold: 0, n_particles: 10, max_iter: 10, ...meta.defaults, max_iter: 10, race: true,
    manualOn: true, startOn: true, manual: { n_estimators: 60, max_depth: 3, min_samples_split: 10 }, seed: null, ...saved() };
  if (st.startOn == null) st.startOn = true;
  let stop = null, running = false, disposed = false;

  // ---------------------------------------------------------------- controls
  const dsCards = h("div", { class: "ds-cards" });
  const drawDs = () => dsCards.replaceChildren(...meta.datasets.map((d) => {
    const b = h("button", { type: "button", class: `ds-card ${st.dataset === d.key ? "on" : ""}` },
      h("b", {}, d.label), h("span", {}, `${d.n_samples} samples · ${d.n_features} features · ${d.n_classes} classes`));
    b.addEventListener("click", () => { if (running) return; st.dataset = d.key; drawDs(); updateEst(); idle(); });
    return b;
  }));
  drawDs();
  const foldSeg = segmented([0, 1, 2, 3, 4].map((k) => [k, `fold ${k}`]), st.fold, (v) => { st.fold = v; updateEst(); idle(); });
  const sN = slider({ label: "Particles N", min: 4, max: 30, value: st.n_particles, onInput: (v) => { st.n_particles = v; updateEst(); },
    hint: "Swarm size: each particle is one candidate configuration." });
  const sT = slider({ label: "Iterations T", min: 2, max: 30, value: st.max_iter, onInput: (v) => { st.max_iter = v; updateEst(); },
    hint: "Swarm updates after the initial evaluation. Budget = N × (T + 1)." });
  const sW = slider({ label: "Inertia w", min: 0.05, max: 1.2, step: 0.01, value: st.w, format: (v) => v.toFixed(2), onInput: (v) => { st.w = v; },
    hint: "How much of the previous velocity is kept. High → explores, low → settles." });
  const sC1 = slider({ label: "Cognitive c₁", min: 0, max: 3, step: 0.01, value: st.c1, format: (v) => v.toFixed(2), onInput: (v) => { st.c1 = v; },
    hint: "Pull toward each particle's own best (pbest)." });
  const sC2 = slider({ label: "Social c₂", min: 0, max: 3, step: 0.01, value: st.c2, format: (v) => v.toFixed(2), onInput: (v) => { st.c2 = v; },
    hint: "Pull toward the swarm's best (gbest)." });
  const sV = slider({ label: "Velocity clamp", min: 0.02, max: 1, step: 0.01, value: st.v_max_frac, format: (v) => `${(v * 100).toFixed(0)}% of range`,
    onInput: (v) => { st.v_max_frac = v; }, hint: "Largest step per iteration, per dimension." });
  const seedIn = h("input", { type: "number", min: 0, max: 1000000, value: st.seed ?? "", placeholder: "= fold", style: { width: "100%" } });
  seedIn.addEventListener("input", () => { st.seed = seedIn.value === "" ? null : Number(seedIn.value); });
  const resetCoef = h("button", { type: "button", class: "btn ghost" }, "Reset to design values");
  resetCoef.addEventListener("click", () => {
    Object.assign(st, meta.defaults); sW.value = st.w; sC1.value = st.c1; sC2.value = st.c2; sV.value = st.v_max_frac;
  });
  const raceT = toggle({ label: "Race random search (same budget, no feedback)", checked: st.race, onChange: (v) => { st.race = v; updateEst(); } });
  const showPick = () => swarm.setManual(st.manualOn || st.startOn ? st.manual : null);
  const pickNote = h("div", { class: "hint" });
  const notePick = () => {
    const c = `(${st.manual.n_estimators}, ${st.manual.max_depth}, ${st.manual.min_samples_split})`;
    pickNote.textContent = st.startOn
      ? `Particle 0 starts at ${c}; the other particles start at random. PSO then tries to improve on your choice.`
      : `The swarm starts fully at random${st.manualOn ? `; ${c} is only scored for comparison` : ""}.`;
  };
  const startT = toggle({ label: "Start the swarm from these values (particle 0)", checked: st.startOn,
    onChange: (v) => { st.startOn = v; showPick(); notePick(); } });
  const manT = toggle({ label: "Also score them on the test fold (comparison)", checked: st.manualOn,
    onChange: (v) => { st.manualOn = v; showPick(); notePick(); } });
  const LBL = { n_estimators: "n_estimators (trees)", max_depth: "max_depth (tree depth)", min_samples_split: "min_samples_split" };
  const manS = HP.map((k, i) => slider({ label: LBL[k], min: [50, 2, 2][i], max: [200, 20, 10][i], value: st.manual[k],
    onInput: (v) => { st.manual[k] = v; showPick(); notePick(); } }));
  notePick();
  const est = h("div", { class: "caption" });
  const runBtn = h("button", { type: "button", class: "btn primary big" }, "▶  Run the closed loop");
  runBtn.addEventListener("click", () => start());
  function updateEst() {
    const budget = st.n_particles * (st.max_iter + 1), secs = budget * T_EVAL[st.dataset] * (st.race ? 1.35 : 1) + 3;
    est.textContent = `Budget ${budget} evaluations per method · about ${fmt.secs(secs)} on this machine`;
  }
  updateEst();

  const side = h("aside", { class: "lab-side card controls" },
    h("div", { class: "ctl" }, h("label", {}, "Dataset"), dsCards),
    h("div", { class: "ctl" }, h("label", {}, "Outer fold"), foldSeg, h("div", { class: "hint" }, "This fold's test part stays sealed until the search ends.")),
    h("div", { class: "pick" }, h("div", { class: "pick-title" }, "Your starting hyperparameters"), ...manS, startT, manT, pickNote),
    sN, sT,
    h("details", { class: "more" }, h("summary", {}, "PSO coefficients"), h("div", { class: "controls" }, sW, sC1, sC2, sV,
      h("div", { class: "ctl" }, h("label", {}, "Run seed"), seedIn, h("div", { class: "hint" }, "Drives PSO, inner folds and forest seeds.")), resetCoef)),
    raceT,
    est, runBtn);

  // ---------------------------------------------------------------- stage
  const viewport = h("div", { class: "viewport" });
  const hud = h("div", { class: "hud" });
  const hudIt = h("div", { class: "big" }), hudG = h("div", { class: "big" }), hudE = h("div", { class: "big" });
  hud.append(hudIt, hudG, hudE);
  const overlay = h("div", { class: "overlay-msg" });
  const chips = h("div", { class: "hud-right" });
  const vis = { trails: true, velocity: true, explored: true, random: true };
  for (const [key, label] of [["trails", "trails"], ["velocity", "velocity"], ["explored", "explored configs"], ["random", "random search"]]) {
    const c = h("button", { type: "button", class: "chip btn-chip on" }, label);
    c.addEventListener("click", () => { vis[key] = !vis[key]; c.classList.toggle("on", vis[key]); swarm.setVisible({ [key]: vis[key] }); });
    chips.append(c);
  }
  const legend = h("div", { class: "legend-3d" },
    h("span", {}, h("i", { style: { background: COLORS.pso } }), "particle (color = fitness)"),
    h("span", {}, h("i", { style: { background: COLORS.gbest } }), "gbest"),
    h("span", {}, h("i", { style: { background: COLORS.random_search } }), "random-search sample"),
    h("span", {}, h("i", { style: { background: "transparent", border: "1.5px dashed #c3c2b7" } }), "your pick"));
  viewport.append(hud, overlay, chips, legend);
  const swarm = new Swarm3D(viewport);
  showPick();

  const loopBox = h("div");
  const loop = new LoopDiagram(loopBox);
  const vault = h("div", { class: "vault" });
  const drawVault = (open, detail) => {
    vault.className = `vault ${open ? "open" : ""}`;
    vault.replaceChildren(h("div", { class: "row" },
      h("span", { html: `<svg class="lock" viewBox="0 0 52 52" aria-hidden="true"><path class="shackle" d="M16 24v-8a10 10 0 0 1 20 0v8" fill="none" stroke="${open ? "#5fd35f" : "#e66767"}" stroke-width="4" stroke-linecap="round"/><rect x="10" y="24" width="32" height="22" rx="5" fill="${open ? "#5fd35f" : "#e66767"}"/><circle cx="26" cy="35" r="3.5" fill="#1a1a19"/></svg>` }),
      h("div", {}, h("div", { class: "k" }, "held-out test fold"), h("div", { class: "t" }, open ? "UNSEALED · scored once" : "SEALED"),
        h("div", { class: "d" }, detail || "Unreachable while any search runs: reveal() raises TestSetAccessError."))));
  };
  drawVault(false);
  const psoBar = h("div", { class: "progress" }, h("i")), rsBar = h("div", { class: "progress rs" }, h("i"));
  const psoLbl = h("div", { class: "lbl" }), rsLbl = h("div", { class: "lbl" });
  const raceCard = h("div", { class: "card race-bars" }, h("h3", {}, "Progress"), h("div", {}, psoLbl, psoBar), h("div", { class: "rsrow" }, rsLbl, rsBar));
  const right = h("div", { class: "grid" }, h("div", { class: "card" }, h("h3", {}, "The loop, live"), loopBox), vault, raceCard);

  const kIt = kpi("iteration", "–", "0 = initial swarm"), kG = kpi("gbest fitness", "–", "validation accuracy (inner CV)");
  const kCfg = kpi("gbest config", "–", "(n_estimators, max_depth, split)"), kDiv = kpi("swarm diversity", "–", "spread of the particles");
  const kEv = kpi("evaluations", "–", "unique Random Forest fits");
  const kpis = h("div", { class: "kpis" }, kIt, kG, kCfg, kDiv, kEv);

  const convBox = h("div"), raceBox = h("div");
  const conv = new LineChart(convBox, { xLabel: "PSO iteration", yLabel: "validation accuracy", integerX: true, height: 240 });
  const race = new LineChart(raceBox, { xLabel: "configurations evaluated", yLabel: "best so far", height: 240 });
  const feed = h("div", { class: "feed" });
  const resultBox = h("div", { class: "card hidden" });
  const status = h("div", { class: "caption" });

  const main = h("div", { class: "lab-main" },
    h("div", { class: "stage-row" }, viewport, right), kpis,
    h("div", { class: "grid cols-2" },
      h("div", { class: "card" }, h("h3", {}, "Convergence", h("span", { class: "sub" }, "gbest is what PSO remembers; the band is this iteration's swarm")), convBox),
      h("div", { class: "card" }, h("h3", {}, "Closed loop vs open loop", h("span", { class: "sub" }, "best validation accuracy so far, same budget")), raceBox)),
    h("div", { class: "grid cols-2" },
      h("div", { class: "card" }, h("h3", {}, "Evaluation feed", h("span", { class: "sub" }, "every configuration sent to the forest")), feed, status),
      resultBox));

  root.append(h("div", { class: "view-head" }, h("div", {},
    h("h1", {}, "Live closed-loop lab"),
    h("p", {}, "PSO proposes Random Forest hyperparameters, a forest is trained, and its validation accuracy is fed back so PSO can move the swarm. This runs the experiment's own code. Tune the swarm, press run, and watch the loop close. The held-out test fold opens only at the end."))),
  h("div", { class: "lab" }, side, main));

  // ---------------------------------------------------------------- run state
  let run = null;
  function idle() {
    if (running) return;
    const d = meta.datasets.find((x) => x.key === st.dataset);
    overlay.innerHTML = `<div><b>${d.label}</b>, outer fold ${st.fold}: <b>${d.n_opt}</b> samples for optimization, <b>${d.n_test}</b> sealed for the final test.<br>Press <b>Run the closed loop</b>. Drag to rotate; Ctrl + scroll (or the + / − buttons) to zoom.</div>`;
    overlay.classList.remove("hidden");
    hudIt.innerHTML = ""; hudG.innerHTML = ""; hudE.innerHTML = "";
  }
  idle();
  function resetRun(params, budget) {
    swarm.reset(); showPick();
    conv.clearAll(); race.clearAll(); feed.replaceChildren(); resultBox.classList.add("hidden"); resultBox.replaceChildren();
    run = { params, budget, pso: [], rs: [], iters: [], best: { pso: -Infinity, random_search: -Infinity }, n: { pso: 0, random_search: 0 },
      unique: 0, lastPulse: 0 };
    conv.set("band", { color: COLORS.pso, band: [], label: "swarm min–max" });
    conv.set("mean", { color: COLORS.ink2, dash: "5 4", points: [], label: "swarm mean", width: 1.5 });
    conv.set("gbest", { color: COLORS.pso, points: [], step: true, dots: true, label: "gbest", width: 2.5 });
    if (params.race) race.set("random_search", { color: COLORS.random_search, points: [], step: true, label: LABEL.random_search, head: true, width: 2.5, dash: "7 4" });
    race.set("pso", { color: COLORS.pso, points: [], step: true, label: LABEL.pso, head: true, width: 2.5 });
    race.o.xMax = budget; conv.o.xMax = params.max_iter;
    raceCard.querySelector(".rsrow").classList.toggle("hidden", !params.race);
    drawVault(false); overlay.classList.add("hidden");
    loop.setValues({ iter: 0, total: params.max_iter });
    progress();
  }
  function progress() {
    psoBar.firstChild.style.transform = `scaleX(${run.n.pso / run.budget})`;
    rsBar.firstChild.style.transform = `scaleX(${run.n.random_search / run.budget})`;
    psoLbl.replaceChildren(h("span", {}, "PSO (closed loop)"), h("span", { class: "mono" }, `${run.n.pso} / ${run.budget} · best ${fmt.acc(run.best.pso > -Infinity ? run.best.pso : null)}`));
    rsLbl.replaceChildren(h("span", {}, "Random search (open loop)"), h("span", { class: "mono" }, `${run.n.random_search} / ${run.budget} · best ${fmt.acc(run.best.random_search > -Infinity ? run.best.random_search : null)}`));
  }
  function feedRow(ev, isBest) {
    const row = h("div", { class: `row ${ev.m} ${isBest ? "best" : ""}` },
      h("span", { class: "m" }, ev.m === "pso" ? `PSO p${ev.p}` : `RS #${ev.i + 1}`), h("span", {}, fmt.cfg(ev.cfg)),
      h("span", {}, fmt.acc(ev.f)), h("span", { class: "tag" }, ev.cache ? "cached" : isBest ? "★ best" : ""));
    feed.prepend(row);
    while (feed.children.length > 14) feed.lastChild.remove();
  }
  function onEvent(ev) {
    if (disposed) return;
    if (ev.t === "start") {
      resetRun(ev.params, ev.budget); run.nTest = ev.n_test; run.classes = ev.classes;
      status.textContent = `Streaming from the experiment's run_fold · saved to ${ev.out}`;
      return;
    }
    if (!run) return;
    if (ev.t === "eval") {
      const m = ev.m, isBest = ev.f != null && ev.f > run.best[m];
      if (isBest) run.best[m] = ev.f;
      run.n[m] += 1;
      if (!ev.cache) run.unique += m === "pso" ? 1 : 0;
      race.append(m, [run.n[m], ev.best]);
      if (m === "pso") {
        swarm.setParticle(ev.p, ev.pos, ev.vel, { ...ev, improved: ev.f === ev.pbest && ev.it > 0 });
        swarm.addExplored(ev.cfg, ev.f);
        loop.setValues({ iter: ev.it, total: run.params.max_iter, particle: ev.p, cfg: ev.cfg, f: ev.f, gcfg: run.gcfg, gf: run.gf });
        if (performance.now() - run.lastPulse > 380) { loop.pulse("forward"); run.lastPulse = performance.now(); }
        hudE.innerHTML = `evaluations <b>${run.n.pso}</b> / ${run.budget}`;
        kEv.set(`${run.n.pso}`, `${run.unique} unique RF fits (rest cached)`);
      } else {
        swarm.addRandom(ev.cfg, ev.f);
      }
      if (m === "pso" || run.n.random_search % 2 === 0 || isBest) feedRow(ev, isBest);
      progress();
      return;
    }
    if (ev.t === "iter") {
      run.gcfg = ev.gcfg; run.gf = ev.gf;
      swarm.setGbest(ev.gcfg);
      conv.append("gbest", [ev.it, ev.gf]); conv.append("mean", [ev.it, ev.mean]);
      conv.series.get("band").band.push([ev.it, ev.min, ev.max]); conv.schedule();
      loop.setValues({ iter: ev.it, total: run.params.max_iter, cfg: ev.gcfg, gcfg: ev.gcfg, gf: ev.gf });
      setTimeout(() => loop.pulse("feedback"), 60);
      hudIt.innerHTML = `iteration <b>${ev.it}</b> / ${run.params.max_iter}`;
      hudG.innerHTML = `gbest <b>${fmt.acc(ev.gf)}</b> ${fmt.cfg(ev.gcfg)}`;
      kIt.set(`${ev.it} / ${run.params.max_iter}`, ev.improved ? "gbest improved ★" : "no gbest change");
      kG.set(fmt.acc(ev.gf), "validation accuracy (inner CV)", ev.improved);
      kCfg.set(fmt.cfg(ev.gcfg), "(n_estimators, max_depth, split)", ev.improved);
      kDiv.set(ev.div == null ? "–" : ev.div.toFixed(2), ev.div < 0.15 ? "converged" : ev.div < 0.35 ? "contracting" : "exploring");
      return;
    }
    if (ev.t === "search_done") { status.textContent = `${LABEL[ev.m]} finished: best ${fmt.cfg(ev.best)} = ${fmt.acc(ev.val)} (validation)`; return; }
    if (ev.t === "phase") { status.textContent = ev.msg; loop.light([]); return; }
    if (ev.t === "vault") { openVault(ev); return; }
    if (ev.t === "cancelled") { status.textContent = "Run cancelled."; finish(); return; }
    if (ev.t === "error") { toast(`Live run failed: ${ev.msg}`, "error", 8000); status.textContent = ev.msg; }
  }
  function openVault(ev) {
    drawVault(true, `${ev.n_test} samples. Each method was scored once, after its own search. Finished in ${fmt.secs(ev.elapsed)}.`);
    const order = ["baseline", "manual", "random_search", "pso"].filter((m) => ev.results[m]);
    const best = Math.max(...order.map((m) => ev.results[m].test.accuracy));
    const one = 1 / ev.n_test;
    const table = h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, "method"), h("th", {}, "configuration"),
      h("th", { class: "num" }, "validation (biased)"), h("th", { class: "num" }, "test accuracy"), h("th", { class: "num" }, "macro F1"), h("th", { class: "num" }, "configs"))),
      h("tbody", {}, ...order.map((m) => { const r = ev.results[m]; return h("tr", {},
        h("td", {}, h("span", { class: `sw ${m === "manual" ? "you" : ""}`, style: { background: COLORS[m] } }), LABEL[m]),
        h("td", { class: "mono" }, fmt.cfg(r.best_hyperparameters)),
        h("td", { class: "num" }, fmt.acc(r.best_validation_fitness)),
        h("td", { class: "num", style: { fontWeight: r.test.accuracy === best ? 700 : 400 } }, fmt.acc(r.test.accuracy)),
        h("td", { class: "num" }, fmt.acc(r.test.f1_macro)), h("td", { class: "num" }, r.n_evaluations)); })));
    const bars = h("div");
    resultBox.replaceChildren(h("h3", {}, "After the loop: the test fold is opened"), h("div", { class: "table-scroll" }, table), bars,
      h("div", { class: "insight", style: { marginTop: "10px" } },
        h("b", {}, "Reading this honestly. "), `On this fold one test sample is worth ${one.toFixed(3)} accuracy, so smaller gaps are noise. `,
        "Validation accuracy is the best of many noisy estimates, so it is optimistically biased; only the test column is performance. ",
        "The Results view averages all 5 outer folds."));
    resultBox.classList.remove("hidden");
    barChart(bars, [{ label: "test accuracy on this fold", bars: order.map((m) => ({ key: m, label: LABEL[m], value: ev.results[m].test.accuracy,
      color: COLORS[m], dashed: m === "manual" })) }], { height: 220 });
    toast("Search finished: the test fold is now open.");
    finish();
  }
  function finish() { running = false; runBtn.disabled = false; runBtn.textContent = "▶  Run again"; stop && stop(); stop = null; }
  async function start() {
    if (running) return;
    save({ ...st });
    const body = { dataset: st.dataset, fold: st.fold, n_particles: st.n_particles, max_iter: st.max_iter, w: st.w, c1: st.c1, c2: st.c2,
      v_max_frac: st.v_max_frac, seed: st.seed, race: st.race, manual: st.manualOn || st.startOn ? st.manual : null,
      start_from_manual: st.startOn, compare_manual: st.manualOn };
    running = true; runBtn.disabled = true; runBtn.textContent = "Running the loop…";
    try {
      const job = await api.startLive(body);
      stop = streamLive(job.id, onEvent, () => { if (running) finish(); },
        () => { toast("This run is no longer on the server (it was restarted or cancelled). Press Run again.", "error", 7000); finish(); });
    } catch (err) { toast(err.message, "error", 7000); running = false; runBtn.disabled = false; runBtn.textContent = "▶  Run the closed loop"; }
  }

  return () => { disposed = true; stop && stop(); swarm.dispose(); conv.destroy(); race.destroy(); };
}
