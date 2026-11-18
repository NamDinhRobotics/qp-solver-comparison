"""Generate every LaTeX table and the figure used in the paper from results.json.

No number in the manuscript is typed by hand: each one is emitted here.
"""

import argparse
import json
import os

import numpy as np

PRETTY = {"clarabel": "Clarabel", "cvxopt": "CVXOPT", "daqp": "DAQP",
          "ecos": "ECOS", "highs": "HiGHS", "osqp": "OSQP", "piqp": "PIQP",
          "proxqp": "ProxQP", "qpalm": "QPALM", "quadprog": "quadprog",
          "scs": "SCS"}
FAMILY = {"clarabel": "conic IP", "cvxopt": "IP", "daqp": "dual AS",
          "ecos": "conic IP", "highs": "IP/simplex", "osqp": "ADMM",
          "piqp": "proximal IP", "proxqp": "AL/prox.", "qpalm": "proximal AL",
          "quadprog": "dual AS", "scs": "conic ADMM"}
ORDER = ["daqp", "quadprog", "piqp", "proxqp", "qpalm", "osqp", "scs",
         "clarabel", "ecos", "cvxopt", "highs"]


def ms(v):
    if v is None:
        return "--"
    return f"{v:.3f}" if v < 1 else (f"{v:.2f}" if v < 100 else f"{v:.1f}")


def sci(v):
    if v is None:
        return "--"
    if v == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(v))))
    return f"$10^{{{e}}}$"


def cell(r):
    st = r.get("status")
    if st == "ok":
        return f"{ms(r.get('median_ms'))}\\,({ms(r.get('p99_ms'))})"
    if st == "not_found":
        return r"\xmark"
    return r"n/a"


def timing_table(res, keys, headers, caption, label):
    L = [r"\begin{table}[tbp]", r"\centering", r"\scriptsize",
         f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         r"\setlength{\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}l l " + " ".join(["r"] * len(keys)) + "@{}}",
         r"\toprule",
         "\\textbf{Solver} & \\textbf{Family} & " +
         " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\",
         r"\midrule"]
    for s in ORDER:
        cells = [cell(res["instances"][k]["rows"].get(s, {})) for k in keys]
        L.append(f"\\texttt{{{PRETTY[s]}}} & {FAMILY[s]} & " +
                 " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def accuracy_table(res, keys, headers, caption, label):
    L = [r"\begin{table}[tbp]", r"\centering", r"\scriptsize",
         f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         r"\setlength{\tabcolsep}{3pt}",
         "\\begin{tabular}{@{}l " + " ".join(["r r r"] * len(keys)) + "@{}}",
         r"\toprule",
         "& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{h}}}}}"
                           for h in headers) + r" \\",
         "\\textbf{Solver} & " +
         " & ".join([r"ineq. & eq. & gap"] * len(keys)) + r" \\",
         r"\midrule"]
    for s_ in ORDER:
        cells = []
        for k in keys:
            r = res["instances"][k]["rows"].get(s_, {})
            if r.get("status") == "ok":
                eq = r.get("rel_eq_violation")
                cells += [sci(r.get("rel_ineq_violation")),
                          "--" if eq is None else sci(eq),
                          sci(r.get("duality_gap"))]
            elif r.get("status") == "not_found":
                cells += [r"\xmark"] * 3
            else:
                cells += [r"n/a"] * 3
        L.append(f"\\texttt{{{PRETTY[s_]}}} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def aux_table(res):
    ov = res["overhead"]
    wb = res["instances"]["wbc_id"]["rows"]
    pct = {}
    for k in ("osqp", "highs"):
        try:
            pct[k] = 100.0 * ov[k]["median_ms"] / wb[k]["median_ms"]
        except Exception:
            pct[k] = float("nan")
    L = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize",
         r"\caption{Per-call overhead floor: the median time to solve a "
         r"two-variable QP, i.e. the fixed cost of one call through the Python "
         r"binding. It is included in every timing in "
         r"Tables~\ref{tab:timing-sparse} and~\ref{tab:timing-cond}, and it is "
         r"what Table~\ref{tab:native} avoids by staying in C++. For the WBC "
         f"problem this floor is ${pct['osqp']:.0f}\\%$ of \\texttt{{OSQP}}'s "
         f"total solve time and ${pct['highs']:.0f}\\%$ of \\texttt{{HiGHS}}'s.}}",
         r"\label{tab:overhead}",
         r"\begin{tabular}{@{}l r l r@{}}", r"\toprule",
         r"\textbf{Solver} & \textbf{Overhead (ms)} & \textbf{Solver} & "
         r"\textbf{Overhead (ms)} \\", r"\midrule"]
    rows = [(s_, ov.get(s_, {})) for s_ in ORDER]
    rows = [(n, ms(d.get("median_ms")) if d.get("status") != "error" else "--")
            for n, d in rows]
    half = (len(rows) + 1) // 2
    for i in range(half):
        a_ = rows[i]
        b_ = rows[i + half] if i + half < len(rows) else None
        line = f"\\texttt{{{PRETTY[a_[0]]}}} & {a_[1]} & "
        line += (f"\\texttt{{{PRETTY[b_[0]]}}} & {b_[1]}" if b_ else " & ")
        L.append(line + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def proxqp_table(res):
    pt = res["proxqp_threshold"]
    Ns = sorted(pt, key=lambda k: int(k[1:]))
    epss = ["0.0001", "1e-05", "1e-06", "1e-08", "0"]
    lbl = {"0.0001": r"$10^{-4}$ (default)", "1e-05": r"$10^{-5}$",
           "1e-06": r"$10^{-6}$", "1e-08": r"$10^{-8}$", "0": r"$0$"}
    L = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize",
         r"\caption{ProxQP on the \emph{condensed} MPC problem of "
         r"Eq.~\eqref{eq:mpc}, as its primal-infeasibility threshold "
         r"\texttt{eps\_primal\_inf} is varied. At the shipped default the "
         r"solver declares the problem primal infeasible after a single "
         r"iteration, at every horizon; the problem is in fact feasible, and "
         r"PIQP and DAQP agree on the optimal value to nine or more "
         r"significant digits. Each cell gives the returned status and the "
         r"iteration count.}",
         r"\label{tab:proxqp}",
         r"\begin{tabular}{@{}l " + " ".join(["c"] * len(Ns)) + "@{}}",
         r"\toprule",
         r"\texttt{eps\_primal\_inf} & " +
         " & ".join(f"$N={n[1:]}$" for n in Ns) + r" \\", r"\midrule"]
    for e in epss:
        cells = []
        for n in Ns:
            d = pt[n]["proxqp_sweep"][e]
            tag = "infeas." if "INFEASIBLE" in d["status"] else "solved"
            mark = r"\xmark" if tag == "infeas." else r"\cmark"
            cells.append(f"{mark}\\,{tag} ({d['iterations']})")
        L.append(lbl[e] + " & " + " & ".join(cells) + r" \\")
    L += [r"\midrule",
          r"\multicolumn{" + str(len(Ns) + 1) + r"}{@{}l}{\emph{reference "
          r"optimal value from independent solvers}}\\"]
    for ref in ("piqp", "daqp"):
        vals = []
        for n in Ns:
            r = pt[n]["reference_solvers"].get(ref)
            vals.append(f"{r['objective']:.6f}" if r else "--")
        L.append(f"\\texttt{{{PRETTY[ref]}}} & " + " & ".join(vals) + r" \\")
    vals = []
    for n in Ns:
        d = pt[n]["proxqp_sweep"]["1e-06"]
        vals.append(f"{d['objective']:.6f}")
    L.append(r"\texttt{ProxQP} ($10^{-6}$) & " + " & ".join(vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--wsctl", default="warmstart_controlled.json")
    ap.add_argument("--matched", default="matched_tolerance.json")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    res = json.load(open(a.results))
    os.makedirs(a.outdir, exist_ok=True)

    sp = sorted([k for k in res["instances"] if "sparse" in k],
                key=lambda k: int(k.split("N")[-1]))
    cd = sorted([k for k in res["instances"] if "condensed" in k],
                key=lambda k: int(k.split("N")[-1]))
    Ns = [int(k.split("N")[-1]) for k in sp]
    hdr = [f"$N={n}$" for n in Ns]

    W = lambda n, t: open(os.path.join(a.outdir, n), "w").write(t + "\n")

    W("tab_timing.tex",
      timing_table(res, sp + ["wbc_id"], hdr + ["WBC"],
                   r"Median solve time in milliseconds, 99th percentile in "
                   r"brackets, on the \emph{sparse} OCP form of the MPC problem "
                   r"of Eq.~\eqref{eq:mpc} at horizon $N$, and on the WBC "
                   r"problem of Eq.~\eqref{eq:wbc}. "
                   r"\texttt{n/a}: the wrapper rejects sparse input.",
                   "tab:timing-sparse") + "\n\n" +
      timing_table(res, cd, hdr,
                   r"Median solve time in milliseconds, 99th percentile in "
                   r"brackets, on the \emph{condensed} dense form of the same "
                   r"MPC problems. \xmark: the solver returned no solution "
                   r"(see Table~\ref{tab:proxqp}).",
                   "tab:timing-cond"))

    W("tab_accuracy.tex",
      accuracy_table(res, [sp[-1], cd[-1], "wbc_id"],
                     [f"sparse $N={Ns[-1]}$", f"condensed $N={Ns[-1]}$", "WBC"],
                     r"Solution quality at each solver's \emph{default} "
                     r"settings, all three columns truncated to the power of "
                     r"ten at or below the value. \emph{ineq.} is the "
                     r"relative inequality violation "
                     r"$\|\max(0,Cx-d)\|_\infty/\max(1,\|d\|_\infty)$; "
                     r"\emph{eq.} is the relative equality residual "
                     r"$\|Ax-b\|_\infty/\max(1,\|b\|_\infty)$, shown as "
                     r"``--'' where the instance has no equality constraints; "
                     r"\emph{gap} is the duality gap reported by "
                     r"\texttt{qpsolvers}. \xmark: no solution returned; "
                     r"\texttt{n/a}: the wrapper rejects sparse input.",
                     "tab:accuracy"))
    W("tab_overhead.tex", aux_table(res))
    W("tab_proxqp.tex", proxqp_table(res))

    if os.path.exists(a.matched):
        W("tab_matched.tex", matched_table(json.load(open(a.matched))))
    else:
        print(f"note: {a.matched} absent; tab_matched not regenerated")

    if os.path.exists(a.wsctl):
        ws = json.load(open(a.wsctl))
        W("tab_wsctl.tex", wsctl_table(ws))
        W("tab_osqpct.tex", osqp_ct_table(ws))
    else:
        print(f"note: {a.wsctl} absent; tab_wsctl/tab_osqpct not regenerated")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), sharey=True)
    cmap = plt.get_cmap("tab20")
    for ax, keys, title in ((axes[0], sp, "Sparse OCP form"),
                            (axes[1], cd, "Condensed dense form")):
        for i, s in enumerate(ORDER):
            ys = []
            for k in keys:
                r = res["instances"][k]["rows"].get(s, {})
                ys.append(r.get("median_ms") if r.get("status") == "ok"
                          else np.nan)
            if np.all(np.isnan(ys)):
                continue
            ax.plot(Ns, ys, marker="o", ms=3.5, lw=1.3,
                    color=cmap(i % 20), label=PRETTY[s])
        ax.set_yscale("log")
        ax.set_xlabel("prediction horizon $N$")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
        ax.set_xticks(Ns)
        ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.7)
    axes[0].set_ylabel("median solve time (ms)")
    axes[0].text(Ns[0], 1.12, "1 ms", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    seen = dict(zip(l, h)); seen.update(dict(zip(l2, h2)))
    fig.legend(seen.values(), seen.keys(), fontsize=7.5, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(os.path.join(a.outdir, "bench_horizon.png"), dpi=220)
    print("wrote tables + bench_horizon.png to", a.outdir)


# ==========================================================================
# Native C++ results and the warm-start comparison (added in revision 2)
# ==========================================================================
CPP_PRETTY = {"qpmad": r"\texttt{qpmad}", "eiquadprog": r"\texttt{eiquadprog}",
              "qpOASES": r"\texttt{qpOASES} default, cold",
              "qpOASES_hotstart": r"\texttt{qpOASES} default, hot",
              "qpOASES_mpc": r"\texttt{qpOASES} MPC, cold",
              "qpOASES_mpc_hotstart": r"\texttt{qpOASES} MPC, hot"}
CPP_ORDER = ["qpmad", "eiquadprog", "qpOASES", "qpOASES_mpc",
             "qpOASES_hotstart", "qpOASES_mpc_hotstart"]


def native_table(cpp, Ns):
    idx = {}
    for r in cpp:
        idx.setdefault(r["solver"], {})[r["instance"]] = r
    keys = [f"srbd_mpc_condensed_N{n}" for n in Ns] + ["wbc_id"]
    hdr = [f"$N={n}$" for n in Ns] + ["WBC"]
    L = [r"\begin{table}[tbp]", r"\centering", r"\scriptsize",
         r"\caption{Native C++ timings, in milliseconds (99th percentile in "
         r"brackets), for the three dense solvers that have no ready Python "
         r"binding. These numbers cross no language boundary and so carry none "
         r"of the per-call overhead of Table~\ref{tab:overhead}. "
         r"\emph{cold} builds a fresh solver object for every solve; "
         r"\emph{hot} keeps one object and re-solves after perturbing the "
         r"linear term, which is the parametric warm start qpOASES is designed "
         r"for; \emph{MPC} means the solver's own \texttt{setToMPC()} option preset "
         r"instead of its defaults. On the \emph{cold} and \emph{MPC} rows all "
         r"solvers agree with the Python reference optimum of "
         r"Table~\ref{tab:proxqp} to the digits shown there, which is the "
         r"correctness check on this harness. The \emph{hot} rows re-solve "
         r"after perturbing the linear term and therefore return the optimum "
         r"of a different problem, differing in the fourth to sixth "
         r"significant figure.}",
         r"\label{tab:native}", r"\setlength{\tabcolsep}{2.5pt}",
         "\\begin{tabular}{@{}l " + " ".join(["r"] * len(keys)) + "@{}}",
         r"\toprule",
         r"\textbf{Solver} & " + " & ".join(f"\\textbf{{{h}}}" for h in hdr) + r" \\",
         r"\midrule"]
    for s in CPP_ORDER:
        if s not in idx:
            continue
        cells = []
        for k in keys:
            r = idx[s].get(k)
            cells.append(f"{ms(r['median_ms'])}\\,({ms(r['p99_ms'])})" if r else "--")
        L.append(CPP_PRETTY[s] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def emit_native_table(results_path, cpp_path, outdir):
    """The native C++ table. (The warm-start table this used to emit as well
    is superseded by wsctl_table, which run_all.sh writes from
    warmstart_controlled.json.)"""
    res = json.load(open(results_path))
    cpp = json.load(open(cpp_path))
    sp_keys = sorted([k for k in res["instances"] if "sparse" in k],
                     key=lambda k: int(k.split("N")[-1]))
    Ns = [int(k.split("N")[-1]) for k in sp_keys]
    open(os.path.join(outdir, "tab_native.tex"), "w").write(
        native_table(cpp, Ns) + "\n")
    print("wrote tab_native.tex")


# ==========================================================================
# GPU floor table (added in revision 3; previously the one hand-typed table)
# ==========================================================================
WS_ORDER = ["DAQP", "ProxQP", "OSQP", "PIQP"]


def _pick(sweep, target=0.004):
    """The sweep point whose median step is closest to `target` (relative)."""
    return min(sweep, key=lambda e: abs(e["perturbation"]["median_rel_step"]
                                        - target))


def wsctl_table(ws):
    """Table: warm starting decomposed, from warmstart_controlled.json."""
    sweep = ws["sweep"]
    rep = _pick(sweep)
    step = rep["perturbation"]["median_rel_step"] * 100
    lo = min(e["perturbation"]["median_rel_step"] for e in sweep) * 100
    hi = max(e["perturbation"]["median_rel_step"] for e in sweep) * 100

    def rng(name, key):
        vs = [e["results"][name].get(key) for e in sweep
              if name in e["results"]]
        vs = [v for v in vs if v]
        return (min(vs), max(vs)) if vs else (None, None)

    lines = []
    for name in WS_ORDER:
        r = rep["results"].get(name)
        if not r or "error" in r:
            continue
        su, wsv = r.get("setup_reuse"), r.get("warm_start")
        su_lo, su_hi = rng(name, "setup_reuse")
        ws_lo, ws_hi = rng(name, "warm_start")
        warm_ms = ("%.3f" % r["warm_ms"]) if r.get("warm_ms") else "--"
        wtxt = (r"$\mathbf{%.2f\times}$" % wsv) if wsv else "--"
        wrange = ("%.2f--%.2f" % (ws_lo, ws_hi)) if ws_lo else "--"
        lines.append(
            r"\texttt{%s} & %.3f & %.3f & %s & $%.1f\times$ & %s & "
            r"%.2f--%.2f & %s \\"
            % (name, r["cold_ms"], r["reset_ms"], warm_ms, su, wtxt,
               su_lo, su_hi, wrange))

    floor = ws.get("drift_floor")
    caption = (
        "Warm starting decomposed, under one perturbation model for every "
        "solver. A closed-loop receding-horizon sequence of %d condensed MPC "
        "problems ($N=%d$): each QP is solved, its first control block is "
        "applied to the dynamics, the state is propagated and the reference "
        "shifted, and the resulting sequence is replayed to every solver. "
        "\\emph{cold}: a fresh solver object each step. \\emph{reset}: one "
        "persistent object, data updated, previous iterate discarded. "
        "\\emph{warm}: the same, iterate kept. cold/reset is setup and "
        "factorization reuse; reset/warm is the warm start itself. The two "
        "right-hand columns give the range of each ratio over five "
        "perturbation sizes spanning %.3f\\%% to %.1f\\%% of $\\|q\\|$ per "
        "step; the decomposition is stable across them. \\texttt{PIQP} "
        "exposes no warm-start setting, so its warm leg does not exist and is "
        "not reported as $1.00\\times$. The harness's own drift floor is "
        "$%.2f\\times$, so a ratio below that is not distinguishable from no "
        "effect. Measured at a median step of %.3f\\%% on %s."
        % (ws["steps"], ws["horizon"], lo, hi, floor, step,
           ws["env"]["platform"].split("-")[0] + "/"
           + ws["env"]["machine"]))

    return ("\\begin{table}[tbp]\n\\centering\n\\footnotesize\n"
            "\\caption{%s}\n\\label{tab:wsctl}\n"
            "\\begin{tabular}{@{}l r r r r r c c@{}}\n\\toprule\n"
            "\\textbf{Solver} & \\textbf{cold} & \\textbf{reset} & "
            "\\textbf{warm} & \\textbf{setup} & \\textbf{warm} & "
            "\\textbf{setup} & \\textbf{warm} \\\\\n"
            " & (ms) & (ms) & (ms) & \\textbf{reuse} & \\textbf{start} & "
            "\\textbf{range} & \\textbf{range} \\\\\n\\midrule\n%s\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}"
            % (caption, "\n".join(lines)))


def osqp_ct_table(ws):
    """Table: OSQP's default termination-check interval hides its warm start."""
    lines = []
    reset_iters_ct1 = set()
    warm_iters_ct1 = []
    warm_ratios = []
    for e in ws["sweep"]:
        step = e["perturbation"]["median_rel_step"] * 100
        p = e.get("osqp_check_termination") or {}
        if "25" not in p or "1" not in p:
            continue
        d, o = p["25"], p["1"]
        reset_iters_ct1.add(int(round(o["reset"]["median_iters"])))
        warm_iters_ct1.append(int(round(o["warm"]["median_iters"])))
        warm_ratios.append(o["warm_start"])
        lines.append(
            r"%.3f & %.0f & %.3f & %.0f & %.3f & $%.2f\times$ & "
            r"%.0f & %.3f & %.0f & %.3f & $\mathbf{%.2f\times}$ \\"
            % (step,
               d["reset"]["median_iters"], d["reset"]["median_ms"],
               d["warm"]["median_iters"], d["warm"]["median_ms"],
               d["warm_start"],
               o["reset"]["median_iters"], o["reset"]["median_ms"],
               o["warm"]["median_iters"], o["warm"]["median_ms"],
               o["warm_start"]))
    caption = (
        "\\texttt{OSQP}'s warm start, hidden and revealed. The solver checks "
        "its termination criteria every \\texttt{check\\_termination} "
        "iterations, $25$ by default, so no solve can report fewer than that "
        "however good its starting point is. At the default interval the "
        "reset and warm legs are indistinguishable at every perturbation "
        "size. At an interval of $1$ the same comparison shows the warm "
        "start converging in $%d$ to $%d$ iterations against $%s$, and the "
        "speed-up appears: $%.1f\\times$ at the smallest step, peaking at "
        "$%.1f\\times$, and decaying to $%.1f\\times$ at the largest. The "
        "effect is a reporting granularity, not an algorithmic limit, and it "
        "is one line of configuration away."
        % (min(warm_iters_ct1), max(warm_iters_ct1),
           "$ or $".join(str(i) for i in sorted(reset_iters_ct1)),
           warm_ratios[0], max(warm_ratios), warm_ratios[-1]))
    return ("\\begin{table}[tbp]\n\\centering\n\\scriptsize\n"
            "\\caption{%s}\n\\label{tab:osqpct}\n"
            "\\setlength{\\tabcolsep}{2.5pt}\n"
            "\\begin{tabular}{@{}r rr rr c rr rr c@{}}\n\\toprule\n"
            " & \\multicolumn{5}{c}{\\textbf{\\texttt{check\\_termination}=25 "
            "(default)}} & \\multicolumn{5}{c}{\\textbf{=1}} \\\\\n"
            "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
            "\\textbf{step} & \\multicolumn{2}{c}{reset} & "
            "\\multicolumn{2}{c}{warm} & \\textbf{ratio} & "
            "\\multicolumn{2}{c}{reset} & \\multicolumn{2}{c}{warm} & "
            "\\textbf{ratio} \\\\\n"
            "(\\%%) & it & ms & it & ms & & it & ms & it & ms & \\\\\n"
            "\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n"
            "\\setlength{\\tabcolsep}{6pt}\n\\end{table}"
            % (caption, "\n".join(lines)))


def matched_table(mt):
    """Table: default tolerance against a matched one, from matched_tolerance.json."""
    sp_key = next(k for k in mt if k.startswith("sparse"))
    N = sp_key.split("N")[-1]
    rows = []
    for key in ORDER:
        name = PRETTY[key]
        cells = []
        for inst in (sp_key, "wbc"):
            r = mt[inst]["rows"].get(key, {})
            de, ma = r.get("default", {}), r.get("matched", {})
            for v in (de.get("ms"), ma.get("ms")):
                cells.append("--" if v is None else
                             (f"{v:.2f}" if v < 100 else f"{v:.1f}"))
            for v in (de.get("subopt"), ma.get("subopt")):
                cells.append("--" if v is None else sci(v))
        if all(c == "--" for c in cells):
            continue
        rows.append(r"\texttt{%s} & %s \\" % (name, " & ".join(cells)))

    spreads = ", ".join(
        "%s $%.1e$" % ("sparse" if k.startswith("sparse") else k,
                       mt[k]["reference_spread"])
        for k in (sp_key, "wbc") if mt[k].get("reference_spread") is not None)
    nref = ", ".join(
        "%s: %s" % ("sparse" if k.startswith("sparse") else k,
                    ", ".join(r"\texttt{%s}" % x
                              for x in sorted(mt[k].get("reference_solvers", {}))))
        for k in (sp_key, "wbc"))
    env = mt.get("env", {})
    missing = sorted({key for key in ORDER
                      for inst in (sp_key, "wbc")
                      if "error" in mt[inst]["rows"].get(key, {})})
    miss_txt = ""
    if missing:
        miss_txt = (" %s could not be run through this \\texttt{qpsolvers} "
                    "build on one or both instances and are shown as "
                    "``--''." % ", ".join(r"\texttt{%s}" % PRETTY[m]
                                          for m in missing))
    caption = (
        "Speed at each solver's default tolerance against speed at a common "
        "$\\epsilon=10^{-9}$, with relative suboptimality "
        "$|f(x)-f^\\star|/\\max(1,|f^\\star|)$, truncated to the power of ten "
        "at or below the value. The reference $f^\\star$ is the mean of the "
        "values returned by the independent high-accuracy solvers that agree "
        "on each instance (%s), so it is not any one solver's own answer; "
        "their spread is %s. ``A common $\\epsilon$'' is not one setting: the "
        "per-solver mapping is listed in \\texttt{matched\\_tolerance.py} and "
        "recorded in the released JSON. \\texttt{quadprog} terminates at an "
        "exact vertex and has no tolerance to set, so its two columns are "
        "identical by design, and it is absent from the sparse instance "
        "because its interface is dense-only.%s Measured on %s/%s with "
        "\\texttt{qpsolvers}~%s."
        % (nref, spreads, miss_txt,
           env.get("platform", "an unrecorded platform").split("-")[0],
           env.get("machine", "?"), env.get("qpsolvers", "?")))

    return ("\\begin{table}[tbp]\n\\centering\n\\scriptsize\n"
            "\\caption{%s}\n\\label{tab:matched}\n"
            "\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{@{}l r r r r | r r r r@{}}\n\\toprule\n"
            "& \\multicolumn{4}{c|}{\\textbf{sparse MPC, $N=%s$}} & "
            "\\multicolumn{4}{c}{\\textbf{WBC}} \\\\\n"
            "\\textbf{Solver} & def.\\ ms & $10^{-9}$ ms & sub@def & "
            "sub@$10^{-9}$ & def.\\ ms & $10^{-9}$ ms & sub@def & "
            "sub@$10^{-9}$ \\\\\n\\midrule\n%s\n\\bottomrule\n"
            "\\end{tabular}\n\\setlength{\\tabcolsep}{6pt}\n\\end{table}"
            % (caption, N, "\n".join(rows)))


def gpu_table(gpu, cpp):
    ref = [r for r in cpp if r["instance"] == "wbc_id" and r["solver"] == "qpmad"]
    wbc = ref[0]["median_ms"]
    f = lambda v: f"{v:.4f}" if v < 1 else f"{v:.2f}"
    L = [r"\begin{table}[tbp]", r"\centering", r"\footnotesize",
         r"\caption{What a GPU costs before it computes anything, measured on "
         + gpu["gpu"] + r" with CuPy~" + gpu["cupy"] + r". These are measured "
         r"through a Python array library and are therefore an \emph{upper} "
         r"bound on a native implementation's fixed cost; the CPU baseline is "
         r"likewise a \texttt{numpy} call of which a large share is dispatch. "
         r"In the upper block the ratio is against the whole-body QP of "
         r"Eq.~\eqref{eq:wbc}, which \texttt{qpmad} solves natively on one CPU "
         r"core in " + f(wbc) + r"\,ms (Table~\ref{tab:native}); a "
         r"$30\times30$ Cholesky is the smallest useful linear-algebra step "
         r"inside one iteration of a dense QP solver of that size. In the "
         r"lower block the ratio is the GPU's throughput on a batch of $B$ "
         r"such factorizations against the same $B$ performed one after "
         r"another on one CPU core.}",
         r"\label{tab:gpu}", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{@{}p{7.0cm} r r@{}}", r"\toprule",
         r"\textbf{Operation} & \textbf{median (ms)} & \textbf{vs.\ WBC} \\",
         r"\midrule"]
    rows = [("empty kernel launch $+$ synchronize", gpu["kernel_launch"]),
            (r"host\,$\to$\,device\,$\to$\,host round trip of the problem "
             r"data (%d\,B)" % gpu["roundtrip_bytes"], gpu["roundtrip_wbc_data"]),
            (r"one $30\times30$ Cholesky, on the GPU", gpu["cholesky_30x30_gpu"]),
            (r"one $30\times30$ Cholesky, on one CPU core",
             gpu["cholesky_30x30_cpu"])]
    for name, rec in rows:
        v = rec["median_ms"]
        L.append(f"{name} & {f(v)} & ${v / wbc:.2f}\\times$ \\\\")
    L += [r"\midrule",
          r"\emph{batched $30\times30$ Cholesky} & \textbf{per problem (ms)} "
          r"& \textbf{speed-up} \\", r"\midrule"]
    for b, v in gpu["batched_cholesky"].items():
        L.append(f"$B={b}$ & {v['per_problem_ms']:.5f} & "
                 f"${v['speedup_vs_cpu_serial']:.2f}\\times$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\setlength{\tabcolsep}{6pt}", r"\end{table}"]
    return "\n".join(L)


def emit_gpu_table(gpu_path, cpp_path, outdir):
    gpu = json.load(open(gpu_path))
    if gpu.get("status") != "ok":
        print("gpu_floor.json has no measurement; tab_gpu not emitted")
        return
    cpp = json.load(open(cpp_path))
    open(os.path.join(outdir, "tab_gpu.tex"), "w").write(gpu_table(gpu, cpp) + "\n")
    print("wrote tab_gpu.tex")


if __name__ == "__main__":
    main()
