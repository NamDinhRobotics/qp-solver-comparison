"""How much of this paper's ordering survives a different random draw?

Every instance elsewhere in the benchmark is generated from a single seed, so
each reported ordering rests on one draw. This script repeats the two
instance families and the warm-start decomposition across several seeds and
reports what moves.

    python seed_study.py --seeds 0,1,2,3,4 --out seed_study.json

Three things are measured per seed:

  wbc        the synthetic whole-body instance of `problems.wbc_id`
  cond_N10   the condensed MPC instance of `problems.srbd_mpc`
  warmstart  the cold/reset/warm decomposition of warmstart_controlled.py,
             at one perturbation size

For the two timing families we record each solver's median and the resulting
ordering, then report, per solver, the range of its rank and the spread of its
median time across seeds. A claim of the form "solver A beats solver B" is
only supported if A's worst rank is still better than B's best.

The sparse $N=40$ family is deliberately excluded: a single sweep over it
costs minutes per seed (HiGHS alone takes ~0.2 s per solve) and the ordering
claims the paper makes about it come from the Linux host, which this script
cannot reach.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import problems  # noqa: E402
import warmstart_controlled as wsc  # noqa: E402


def dense(A):
    return A.toarray() if hasattr(A, "toarray") else np.asarray(A)


def time_instance(d, repeats):
    """Median time per solver on one instance, through qpsolvers."""
    import qpsolvers
    from qpsolvers import Problem, solve_problem
    A = d.get("A")
    prob = Problem(dense(d["P"]), d["q"], dense(d["G"]), d["h"],
                   dense(A) if A is not None else None,
                   d["b"] if A is not None else None)
    rows = {}
    for s in sorted(qpsolvers.available_solvers):
        try:
            for _ in range(3):
                solve_problem(prob, solver=s)
            ts = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                sol = solve_problem(prob, solver=s)
                ts.append((time.perf_counter() - t0) * 1e3)
            if not getattr(sol, "found", False):
                rows[s] = dict(found=False)
                continue
            rows[s] = dict(median_ms=statistics.median(ts), found=True)
        except Exception as exc:
            rows[s] = dict(error=f"{type(exc).__name__}: {exc}")
    return rows


def ranking(rows):
    ok = [(s, r["median_ms"]) for s, r in rows.items() if r.get("found")]
    return [s for s, _ in sorted(ok, key=lambda kv: kv[1])]


def summarise_family(per_seed):
    """Rank range and time spread per solver, across seeds."""
    ranks, times = {}, {}
    for rows in per_seed:
        order = ranking(rows)
        for i, s in enumerate(order):
            ranks.setdefault(s, []).append(i + 1)
        for s, r in rows.items():
            if r.get("found"):
                times.setdefault(s, []).append(r["median_ms"])
    out = {}
    for s in sorted(ranks):
        t = times.get(s, [])
        out[s] = dict(rank_min=min(ranks[s]), rank_max=max(ranks[s]),
                      rank_median=statistics.median(ranks[s]),
                      median_ms=statistics.median(t) if t else None,
                      time_spread=(max(t) / min(t)) if len(t) > 1 else None,
                      n_seeds=len(ranks[s]))
    return out


def separated(summary):
    """Pairs whose rank intervals do not overlap, in median-rank order.

    This is the only ordering statement the data supports: A is above B only
    if A's worst rank over the seeds still beats B's best.
    """
    order = sorted(summary, key=lambda s: summary[s]["rank_median"])
    out = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            if summary[a]["rank_max"] < summary[b]["rank_min"]:
                out.append([a, b])
    return order, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--ws-steps", type=int, default=60)
    ap.add_argument("--ws-sigma", type=float, default=1e-4)
    ap.add_argument("--out", default="seed_study.json")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]

    out = dict(seeds=seeds, repeats=a.repeats,
               ws_sigma=a.ws_sigma, ws_steps=a.ws_steps,
               env=wsc.environment(), families={}, warmstart={})

    for fam, build in (("wbc", lambda s: problems.wbc_id(seed=s)),
                       ("cond_N10", lambda s: problems.srbd_mpc(
                           horizon=10, form="condensed", seed=s))):
        per_seed = []
        print(f"\n=== {fam} ===")
        for s in seeds:
            rows = time_instance(build(s), a.repeats)
            per_seed.append(rows)
            print(f"  seed {s}: {' '.join(ranking(rows))}")
        summary = summarise_family(per_seed)
        order, pairs = separated(summary)
        out["families"][fam] = dict(per_seed=per_seed, summary=summary,
                                    median_order=order,
                                    separated_pairs=pairs)
        print(f"  {'solver':10s} {'rank':>9s} {'median ms':>10s} {'spread':>8s}")
        for s in order:
            v = summary[s]
            rr = (f"{v['rank_min']}" if v["rank_min"] == v["rank_max"]
                  else f"{v['rank_min']}-{v['rank_max']}")
            sp = f"{v['time_spread']:.2f}x" if v["time_spread"] else "--"
            print(f"  {s:10s} {rr:>9s} {v['median_ms']:10.4f} {sp:>8s}")

    # --- the warm-start decomposition, across the same seeds ---
    print("\n=== warm-start decomposition ===")
    ws = {}
    for s in seeds:
        P, G, h, qs, info = wsc.closed_loop_sequence(
            a.ws_steps, 10, seed=s, sigma=a.ws_sigma)
        ref = wsc.exact_reference(P, G, h, qs)
        res, floor = wsc.measure(P, G, h, qs, 1, ref)
        ws[str(s)] = dict(step=info["median_rel_step"], drift=floor,
                          results={k: {kk: v.get(kk) for kk in
                                       ("setup_reuse", "warm_start")}
                                   for k, v in res.items() if "error" not in v})
        print(f"  seed {s} (step {info['median_rel_step']*100:.3f}%): " +
              "  ".join(
                  f"{k} {('%.1fx' % v['setup_reuse']) if v.get('setup_reuse') else '-'}"
                  f"/{('%.2fx' % v['warm_start']) if v.get('warm_start') else 'n/a'}"
                  for k, v in ws[str(s)]["results"].items()))
    agg = {}
    for name in ("DAQP", "ProxQP", "OSQP", "PIQP"):
        for key in ("setup_reuse", "warm_start"):
            vals = [ws[str(s)]["results"].get(name, {}).get(key)
                    for s in seeds]
            vals = [v for v in vals if v]
            if vals:
                agg.setdefault(name, {})[key] = dict(
                    median=statistics.median(vals),
                    min=min(vals), max=max(vals))
    out["warmstart"] = dict(per_seed=ws, summary=agg)
    print(f"\n  {'solver':8s} {'setup reuse':>22s} {'warm start':>20s}")
    for name, v in agg.items():
        f = lambda k: (f"{v[k]['median']:.2f} ({v[k]['min']:.2f}-{v[k]['max']:.2f})"
                       if k in v else "n/a")
        print(f"  {name:8s} {f('setup_reuse'):>22s} {f('warm_start'):>20s}")

    json.dump(out, open(a.out, "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
