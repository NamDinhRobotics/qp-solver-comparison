"""The analysis figures of Sect. 5, generated from the measured JSON files.

Four figures, each answering one question a practitioner actually asks:

  fig_pareto.png   -- what does speed cost in accuracy?
  fig_budget.png   -- which solvers fit inside a 1 kHz / 100 Hz loop?
  fig_defaults.png -- how much is left on the table by defaults and by
                      warm-starting through the wrong interface?
  fig_crossover.png-- at which horizon does condensing stop paying?

Solvers are colored by algorithm family throughout, because the family is
what explains the clustering in every one of them.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PRETTY = {"clarabel": "Clarabel", "cvxopt": "CVXOPT", "daqp": "DAQP",
          "ecos": "ECOS", "highs": "HiGHS", "osqp": "OSQP", "piqp": "PIQP",
          "proxqp": "ProxQP", "qpalm": "QPALM", "quadprog": "quadprog",
          "scs": "SCS", "qpmad": "qpmad", "eiquadprog": "eiquadprog",
          "qpOASES": "qpOASES"}

# algorithm family -> color. One palette for the whole paper.
FAM = {"daqp": "AS", "quadprog": "AS", "qpmad": "AS", "eiquadprog": "AS",
       "qpOASES": "AS",
       "piqp": "IP", "clarabel": "IP", "ecos": "IP", "cvxopt": "IP",
       "highs": "IP",
       "osqp": "OS", "scs": "OS",
       "proxqp": "AL", "qpalm": "AL"}
CFAM = {"AS": "#1f6fb4", "IP": "#e07b16", "OS": "#2e9e4f", "AL": "#8e44ad"}
FAMNAME = {"AS": "active set", "IP": "interior point",
           "OS": "operator splitting", "AL": "augmented Lagrangian"}
EPS = 2.220446049250313e-16


def famcolor(s):
    return CFAM[FAM[s]]


def family_legend(ax, **kw):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=CFAM[f], label=FAMNAME[f])
         for f in ("AS", "IP", "OS", "AL")]
    ax.legend(handles=h, **kw)


# --------------------------------------------------------------------------
def fig_pareto(res, outdir):
    """Solve time against the constraint violation actually attained."""
    panels = [("wbc_id", "Whole-body inverse dynamics  ($n=30$)"),
              ("srbd_mpc_sparse_N40", "Sparse MPC, $N=40$  ($n=1000$)")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    for ax, (key, title) in zip(axes, panels):
        rows = res["instances"][key]["rows"]
        pts = []
        for s, r in rows.items():
            if r.get("status") != "ok":
                continue
            v = r.get("rel_ineq_violation")
            if v is None:
                continue
            v = max(v, EPS / 4)          # plot exact zeros at the floor
            t = r["median_ms"]
            pts.append((v, t, s))
        # place labels alternately above / below to limit collisions
        pts.sort(key=lambda z: z[1])
        for i, (v, t, s) in enumerate(pts):
            ax.scatter(v, t, s=58, color=famcolor(s), zorder=3,
                       edgecolor="white", linewidth=0.8)
            dy = 9 if i % 2 == 0 else -13
            ax.annotate(PRETTY[s], (v, t), textcoords="offset points",
                        xytext=(8, dy), fontsize=7.5, color="#333333",
                        zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("relative constraint violation attained")
        ax.set_title(title, fontsize=9.5)
        ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.6)
        ax.axvline(1e-6, color="#b03030", lw=1.0, ls="--", alpha=0.8)
        ax.set_xlim(2e-18, 3e-1)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo * 0.72, hi * 1.9)
    axes[0].set_ylabel("median solve time (ms)")
    axes[0].text(1.5e-6, axes[0].get_ylim()[1] * 0.55,
                 "looser than $10^{-6}$ $\\rightarrow$", fontsize=7,
                 color="#b03030")
    family_legend(axes[1], fontsize=7.5, loc="upper right", frameon=True,
                  framealpha=0.95)
    fig.suptitle("A loose tolerance buys nothing here: on both problems the "
                 "fastest solver is also an exact one",
                 fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(outdir, "fig_pareto.png"), dpi=220)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_budget(res, cpp, outdir):
    """p99 solve time against real-time budgets, for the WBC problem."""
    rows = []
    for s, r in res["instances"]["wbc_id"]["rows"].items():
        if r.get("status") == "ok":
            rows.append((PRETTY[s], r["p99_ms"], famcolor(s), "Python"))
    idx = {r["solver"]: r for r in cpp if r["instance"] == "wbc_id"}
    for key, label in (("qpmad", "qpmad"), ("eiquadprog", "eiquadprog"),
                       ("qpOASES_mpc", "qpOASES (MPC preset)")):
        if key in idx:
            rows.append((label + "  [C++]", idx[key]["p99_ms"],
                         CFAM["AS"], "C++"))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    y = np.arange(len(rows))
    bars = ax.barh(y, [r[1] for r in rows],
                   color=[r[2] for r in rows],
                   alpha=[0.95 if r[3] == "C++" else 0.62 for r in rows][0]
                   if False else None, zorder=3)
    for b, r in zip(bars, rows):
        b.set_alpha(1.0 if r[3] == "C++" else 0.55)
        b.set_edgecolor(r[2])
        b.set_linewidth(1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("99th-percentile solve time (ms) — the number that sizes a "
                  "control loop")
    for xv, lab, col in ((1.0, "1 kHz budget", "#b03030"),
                         (10.0, "100 Hz budget", "#777777")):
        ax.axvline(xv, color=col, lw=1.4, ls="--", zorder=2)
        ax.text(xv, -1.25, lab, color=col, fontsize=8.5, ha="center",
                va="bottom", fontweight="bold")
    for b, r in zip(bars, rows):
        ax.text(r[1] * 1.12, b.get_y() + b.get_height() / 2,
                f"{r[1]:.3f}" if r[1] < 1 else f"{r[1]:.2f}",
                va="center", fontsize=7.5, color="#333333")
    ax.set_xlim(1e-3, 60)
    ax.set_ylim(len(rows) - 0.3, -1.4)
    ax.grid(True, axis="x", which="both", ls=":", lw=0.5, alpha=0.6)
    family_legend(ax, fontsize=8, loc="upper right", frameon=True,
                  framealpha=0.97, bbox_to_anchor=(0.995, 0.46))
    ax.set_title("Every solver fits a 100 Hz loop; the 1 kHz line is where "
                 "the field thins out\n(solid = native C++, faded = through "
                 "the Python binding)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_budget.png"), dpi=220)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_defaults(res, cpp, wsctl, outdir):
    """Two ways to lose an order of magnitude without changing solver."""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))

    # --- (a) qpOASES: defaults vs its own MPC preset vs hot start
    ax = axes[0]
    Ns = [10, 20, 30, 40]
    idx = {}
    for r in cpp:
        idx.setdefault(r["solver"], {})[r["instance"]] = r["median_ms"]
    series = [("qpOASES", "default options", "#c0392b", "o", "-"),
              ("qpOASES_mpc", "setToMPC() preset", "#e07b16", "s", "-"),
              ("qpOASES_mpc_hotstart", "setToMPC() + hot start",
               "#1f6fb4", "^", "-")]
    for key, lab, col, mk, ls in series:
        ys = [idx[key][f"srbd_mpc_condensed_N{n}"] for n in Ns]
        ax.plot(Ns, ys, marker=mk, ls=ls, color=col, lw=1.8, ms=5, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(Ns)
    ax.set_xlabel("prediction horizon $N$")
    ax.set_ylabel("median solve time (ms)")
    ax.axhline(30.0, color="#777777", ls="--", lw=1.2)
    ax.text(10.2, 34, "30 ms control period", fontsize=7.5, color="#777777")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")
    r40 = idx["qpOASES"]["srbd_mpc_condensed_N40"]
    m40 = idx["qpOASES_mpc"]["srbd_mpc_condensed_N40"]
    ax.annotate("", xy=(40, m40), xytext=(40, r40),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.2))
    ax.text(36.5, np.sqrt(r40 * m40), f"{r40 / m40:.0f}$\\times$",
            fontsize=10, color="#333333", ha="right", fontweight="bold")
    ax.set_title("(a) One solver, one problem, three option sets",
                 fontsize=9.5)

    # --- (b) the warm-start decomposition, from warmstart_controlled.json
    ax = axes[1]
    sweep = wsctl["sweep"]
    rep = min(sweep, key=lambda e: abs(
        e["perturbation"]["median_rel_step"] - 0.004))
    floor = wsctl.get("drift_floor") or 1.0
    key = {"DAQP": "daqp", "ProxQP": "proxqp",
           "OSQP": "osqp", "PIQP": "piqp"}

    names, setup, warm, cols = [], [], [], []
    for n in ("DAQP", "ProxQP", "OSQP", "PIQP"):
        r = rep["results"].get(n)
        if not r or "error" in r:
            continue
        names.append(n)
        setup.append(r.get("setup_reuse") or np.nan)
        warm.append(r.get("warm_start") or np.nan)
        cols.append(famcolor(key[n]))

    x = np.arange(len(names))
    w = 0.36
    floor_patch = ax.axhspan(
        1.0 / floor, floor, color="#dcdcdc", zorder=1, lw=0,
        label=f"within the {floor:.2f}$\\times$ harness noise floor")
    ax.bar(x - w / 2, setup, w, color=cols, edgecolor="white",
           label="setup reuse (cold/reset)", zorder=3)
    ax.bar(x + w / 2, warm, w, color=cols, edgecolor="white", hatch="///",
           alpha=0.5, label="warm start (reset/warm)", zorder=3)

    for xi, v in zip(x, setup):
        if v == v:
            ax.text(xi - w / 2, v * 1.16,
                    (f"{v:.0f}$\\times$" if v >= 10 else f"{v:.2f}$\\times$"),
                    ha="center", fontsize=8, fontweight="bold", zorder=5)
    for xi, v, n in zip(x, warm, names):
        if v == v:
            ax.text(xi + w / 2, v * 1.16, f"{v:.2f}$\\times$", ha="center",
                    fontsize=8, fontweight="bold", zorder=5)
        else:
            ax.text(xi + w / 2, 1.02, "none\noffered", ha="center",
                    va="bottom", fontsize=6.8, color="#777777", zorder=5)

    # what OSQP's default termination-check interval hides
    probe = rep.get("osqp_check_termination") or {}
    if "1" in probe and "OSQP" in names:
        xo = names.index("OSQP")
        v = probe["1"]["warm_start"]
        ax.plot([xo, xo + w], [v, v], color="#c0392b", lw=1.8,
                ls=(0, (4, 2)), zorder=6, solid_capstyle="butt")
        ax.annotate(f"{v:.1f}$\\times$ with the termination\n"
                    "check interval set to 1",
                    xy=(xo + w / 2, v), xytext=(xo + 0.1, v * 2.6),
                    fontsize=6.8, color="#c0392b", ha="center", zorder=6,
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0))

    ax.axhline(1.0, color="#333333", lw=1.0, zorder=2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel("speed-up ($\\times$)")
    ax.set_ylim(0.62, 200)
    ax.grid(True, axis="y", which="major", ls=":", lw=0.5, alpha=0.5)
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.95)
    ax.set_title("(b) Keeping the object, and warm starting, separated",
                 fontsize=9.5)

    fig.suptitle("Two configuration choices worth orders of magnitude",
                 fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(os.path.join(outdir, "fig_defaults.png"), dpi=220)
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_crossover(res, outdir):
    """Best condensed time divided by best sparse time, against horizon."""
    Ns = [10, 20, 30, 40]
    ratio, best_s, best_c = [], [], []
    for n in Ns:
        sp = res["instances"][f"srbd_mpc_sparse_N{n}"]["rows"]
        cd = res["instances"][f"srbd_mpc_condensed_N{n}"]["rows"]
        bs = min(r["median_ms"] for r in sp.values() if r.get("status") == "ok")
        bc = min(r["median_ms"] for r in cd.values() if r.get("status") == "ok")
        best_s.append(bs)
        best_c.append(bc)
        ratio.append(bc / bs)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(Ns, ratio, marker="o", color="#1f6fb4", lw=2.2, ms=7, zorder=3)
    ax.axhline(1.0, color="#333333", lw=1.2)
    ax.fill_between([8, 42], 1, 6, color="#1f6fb4", alpha=0.07)
    ax.fill_between([8, 42], 0.4, 1, color="#e07b16", alpha=0.09)
    ax.text(9.2, 4.6, "sparse OCP form wins", fontsize=10, ha="left",
            color="#1f6fb4", fontweight="bold")
    ax.text(21.0, 0.50, "condensing wins", fontsize=10, color="#c06a10",
            fontweight="bold")
    for n, r, bs, bc in zip(Ns, ratio, best_s, best_c):
        ax.annotate(f"{r:.2f}$\\times$", (n, r), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9,
                    fontweight="bold")
        ax.annotate(f"{bs:.2f} / {bc:.2f} ms", (n, r),
                    textcoords="offset points", xytext=(0, -19), ha="center",
                    fontsize=7.5, color="#555555")
    xc = np.interp(1.0, ratio, Ns)
    ax.plot([xc], [1.0], marker="*", ms=16, color="#c0392b", zorder=4)
    ax.annotate(f"crossover near $N\\approx{xc:.0f}$", (xc, 1.0),
                textcoords="offset points", xytext=(14, -26), fontsize=9,
                color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1))
    ax.set_xticks(Ns)
    ax.set_xlim(8, 42)
    ax.set_ylim(0.42, 7.0)
    ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter
    ax.set_yticks([0.5, 1, 2, 3, 5])
    ax.set_yticklabels(["0.5", "1", "2", "3", "5"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("prediction horizon $N$")
    ax.set_ylabel("best condensed time / best sparse time")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.set_title("\"Exploit the sparsity\" is advice with a horizon attached\n"
                 "(labels: best sparse / best condensed median solve time)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_crossover.png"), dpi=220)
    plt.close(fig)
    return xc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--cpp", default="cpp_results.json")
    ap.add_argument("--wsctl", default="warmstart_controlled.json")
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    res = json.load(open(a.results))
    cpp = json.load(open(a.cpp))

    fig_pareto(res, a.outdir)
    fig_budget(res, cpp, a.outdir)
    wsctl = json.load(open(a.wsctl))
    fig_defaults(res, cpp, wsctl, a.outdir)
    xc = fig_crossover(res, a.outdir)
    print(f"crossover horizon (interpolated): {xc:.2f}")
    print("wrote fig_pareto.png fig_budget.png fig_defaults.png "
          "fig_crossover.png to", a.outdir)


if __name__ == "__main__":
    main()
