// Results explorer: the saved 3 datasets × 5 outer folds × 3 methods experiment (files only).
import { api } from "../api.js";
import { LineChart, barChart, heatmap } from "../charts.js";
import { COLORS, LABEL, fmt, h, kpi } from "../ui.js";

const METHODS = ["baseline", "random_search", "pso"];

export function mount(root, meta) {
  const charts = [], panelCharts = [];
  let disposed = false;
  const body = h("div", { class: "grid" }, h("div", { class: "caption" }, "Loading saved results…"));
  const expSel = h("select", {}, ...meta.experiments.map((e) => h("option", { value: e.id }, `${e.id} · ${e.status}`)));
  if (meta.default_experiment) expSel.value = meta.default_experiment;
  expSel.addEventListener("change", () => load(expSel.value));
  root.append(h("div", { class: "view-head" }, h("div", {}, h("h1", {}, "Experiment results"),
    h("p", {}, "3 datasets × 5 outer folds × {default RF, random search, PSO}. Every sample is tested exactly once, and each method is scored once per fold after its own search. Everything on this page is read from saved result files.")),
    h("div", { class: "ctl" }, h("label", {}, "Results directory"), expSel)), body);

  async function load(exp) {
    [...charts.splice(0), ...panelCharts.splice(0)].forEach((c) => c.destroy && c.destroy());
    body.replaceChildren(h("div", { class: "caption" }, "Loading saved results…"));
    const R = await api.results(exp);
    if (disposed) return;
    body.replaceChildren(provenance(R), findings(R), overview(R), datasets(R));
  }

  function provenance(R) {
    const m = R.manifest, ok = !R.problems.length;
    return h("div", { class: "card" }, h("div", { class: "btn-row", style: { alignItems: "center" } },
      h("span", { class: `pill ${ok ? "ok" : "bad"}` }, ok ? "✓ verified: all checks pass" : `✕ ${R.problems.length} verification problem(s)`),
      h("span", { class: `pill ${m.git_dirty === false ? "ok" : "bad"}` }, m.git_dirty === false ? "✓ clean git tree" : "uncommitted changes"),
      h("span", { class: "pill info" }, `commit ${(m.git_commit || "?").slice(0, 8)}`),
      h("span", { class: "pill info" }, `config ${m.config_hash.slice(0, 10)}`),
      h("span", { class: "caption" }, `${m.created_at} → ${m.finished_at} · Python ${m.python_version} · scikit-learn ${m.packages["scikit-learn"]} · ${m.cpu_count} CPUs`)),
      ok ? null : h("ul", {}, ...R.problems.map((p) => h("li", {}, p))),
      h("div", { class: "caption", style: { marginTop: "8px" } }, "The badge re-runs the same audit as python -m pso_rf verify: completeness, test isolation in the run log, and summary recomputation."));
  }

  function row(R, ds, method) { return R.summary.find((r) => r.dataset === ds && r.method === method); }

  function findings(R) {
    const items = Object.keys(R.datasets).map((ds) => {
      const pso = row(R, ds, "pso"), base = row(R, ds, "baseline"), rs = row(R, ds, "random_search");
      const nTest = R.datasets[ds].audit.n_samples / 5, one = 1 / nTest;
      const d = pso.delta_accuracy_mean, verdict = Math.abs(d) < one ? "within one test sample of" : d > 0 ? "above" : "below";
      return h("div", { class: "kpi" }, h("div", { class: "k" }, R.datasets[ds].label),
        h("div", { class: "v", style: { fontSize: "18px" } }, `PSO ${fmt.acc(pso.test_accuracy_mean)}`),
        h("div", { class: "s" }, `default ${fmt.acc(base.test_accuracy_mean)} · random ${fmt.acc(rs.test_accuracy_mean)}`),
        h("div", { class: "s", style: { marginTop: "4px" } }, `PSO is ${verdict} the default (Δ ${fmt.delta(d)}; W/T/L ${pso.wins}/${pso.ties}/${pso.losses}); vs random search ${pso.wins_vs_rs}/${pso.ties_vs_rs}/${pso.losses_vs_rs}.`),
        h("div", { class: "s" }, `Unique RF fits: PSO ${pso.n_unique_fits_mean.toFixed(0)} vs random ${rs.n_unique_fits_mean.toFixed(0)} per fold.`));
    });
    return h("div", { class: "grid" }, h("div", { class: "kpis", style: { gridTemplateColumns: "repeat(auto-fit, minmax(min(300px, 100%), 1fr))" } }, ...items),
      h("div", { class: "insight" }, h("b", {}, "How to read this. "), "Only held-out test accuracy is performance. Validation accuracy is the best of many noisy estimates, so it is optimistically biased. With 5 paired folds no significance test is possible (the smallest two-sided Wilcoxon p-value is 0.0625), so the results are descriptive. A difference smaller than one test sample is noise."));
  }

  function overview(R) {
    const box = h("div");
    const groups = Object.keys(R.datasets).map((ds) => ({
      label: R.datasets[ds].label,
      bars: METHODS.map((m) => ({ key: m, label: LABEL[m], color: COLORS[m], value: row(R, ds, m).test_accuracy_mean,
        dots: R.folds.filter((f) => f.dataset === ds && f.method === m).map((f) => f.test_accuracy) })),
    }));
    const card = h("div", { class: "card" }, h("h3", {}, "Held-out test accuracy", h("span", { class: "sub" }, "bar = mean of 5 outer folds, dots = folds; hover for values")), box,
      h("div", { class: "legend" }, ...METHODS.map((m) => h("span", {}, h("i", { style: { background: COLORS[m], height: "10px", width: "10px" } }), LABEL[m]))));
    requestAnimationFrame(() => charts.push(barChart(box, groups, { height: 300, yLabel: "test accuracy" })));
    return card;
  }

  function datasets(R) {
    const keys = Object.keys(R.datasets), tabs = h("div", { class: "tabs", role: "tablist" }), panel = h("div");
    const show = (ds) => {
      [...tabs.children].forEach((b) => b.classList.toggle("on", b.dataset.ds === ds));
      panelCharts.splice(0).forEach((c) => c.destroy && c.destroy());
      panel.replaceChildren(datasetPanel(R, ds));
    };
    keys.forEach((ds) => { const b = h("button", { type: "button", role: "tab", "data-ds": ds }, R.datasets[ds].label); b.addEventListener("click", () => show(ds)); tabs.append(b); });
    requestAnimationFrame(() => show(keys[keys.length - 1]));
    return h("div", { class: "card" }, tabs, panel);
  }

  function datasetPanel(R, ds) {
    const D = R.datasets[ds];
    const anyBox = h("div"), convBox = h("div");
    const wrap = h("div", { class: "grid" });
    const dep = D.deployment;
    wrap.append(h("div", { class: "kpis" },
      kpi("samples", D.audit.n_samples, `${D.audit.n_features} features · ${D.audit.n_classes} classes`),
      kpi("class balance", D.audit.class_ratio.toFixed(2), "max/min class ratio"),
      kpi("fitness metric", D.audit.fitness_metric, "inner 5-fold CV"),
      dep ? kpi("recommended config", fmt.cfg(dep.recommended_hyperparameters), "deployment run, all data") : null,
      dep && dep.performance_estimate ? kpi("expected test accuracy", fmt.acc(dep.performance_estimate.test_accuracy_mean),
        `± ${fmt.acc(dep.performance_estimate.test_accuracy_std)} (outer-CV estimate)`) : null));
    wrap.append(h("div", { class: "grid cols-2" },
      h("div", {}, h("h3", {}, "Value of feedback", h("span", { class: "sub" }, "best validation accuracy so far · mean of folds, band = min–max")), anyBox),
      h("div", {}, h("h3", {}, "PSO convergence", h("span", { class: "sub" }, "gbest per iteration, one line per outer fold")), convBox)));
    requestAnimationFrame(() => {
      const any = new LineChart(anyBox, { xLabel: "configurations evaluated", yLabel: "best so far", height: 260 });
      for (const m of ["random_search", "pso"]) {
        const a = D.anytime[m]; if (!a) continue;
        any.set(`${m}-band`, { color: COLORS[m], band: a.mean.map((_, i) => [i + 1, a.min[i], a.max[i]]) });
        any.set(m, { color: COLORS[m], points: a.mean.map((v, i) => [i + 1, v]), step: true, width: 2.5, label: LABEL[m] });
      }
      const conv = new LineChart(convBox, { xLabel: "PSO iteration", yLabel: "gbest validation accuracy", integerX: true, height: 260 });
      D.convergence.forEach((c) => conv.set(`f${c.fold}`, { color: COLORS.pso, points: c.gbest.map((v, i) => [i, v]), step: true, width: 1.4, opacity: 0.45, label: `fold ${c.fold}` }));
      const n = D.convergence[0]?.gbest.length || 0;
      conv.set("mean", { color: "#ffffff", points: Array.from({ length: n }, (_, i) => [i, D.convergence.reduce((s, c) => s + c.gbest[i], 0) / D.convergence.length]), step: true, width: 2.5, label: "mean of folds" });
      panelCharts.push(any, conv);
    });
    const tbl = h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, "fold"), h("th", {}, "method"), h("th", {}, "configuration"),
      h("th", { class: "num" }, "validation (biased)"), h("th", { class: "num" }, "test"), h("th", { class: "num" }, "ties with best"), h("th", {}, "on a bound"))),
      h("tbody", {}, ...D.chosen.sort((a, b) => a.fold - b.fold || METHODS.indexOf(a.method) - METHODS.indexOf(b.method)).map((c) => h("tr", {},
        h("td", {}, c.fold), h("td", {}, h("span", { class: "sw", style: { background: COLORS[c.method] } }), LABEL[c.method]),
        h("td", { class: "mono" }, fmt.cfg(c.cfg)), h("td", { class: "num" }, fmt.acc(c.val)), h("td", { class: "num" }, fmt.acc(c.test)),
        h("td", { class: "num" }, c.ties ?? "–"), h("td", {}, (c.bounds || []).join(", ") || "–")))));
    wrap.append(h("div", {}, h("h3", {}, "Chosen configuration per fold", h("span", { class: "sub" }, "many configurations tie at the top: flat landscapes")), h("div", { style: { overflowX: "auto" } }, tbl)));
    const cms = h("div", { class: "grid cols-2" });
    for (const m of ["baseline", "pso"]) {
      const folds = D.chosen.filter((c) => c.method === m);
      if (!folds.length) continue;
      const sum = folds[0].cm.map((r, i) => r.map((_, j) => folds.reduce((s, f) => s + f.cm[i][j], 0)));
      const box = h("div"); heatmap(box, sum, D.audit.class_names, { title: `${LABEL[m]} · pooled over 5 folds (rows = true)` });
      cms.append(box);
    }
    wrap.append(h("div", {}, h("h3", {}, "Pooled confusion matrices", h("span", { class: "sub" }, "every sample tested once · blue = correct, red = errors")), cms));
    return wrap;
  }

  load(expSel.value).catch((err) => body.replaceChildren(h("div", { class: "card" }, String(err.message || err))));
  return () => { disposed = true; [...charts, ...panelCharts].forEach((c) => c.destroy && c.destroy()); };
}
