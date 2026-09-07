#!/usr/bin/env python3
"""Turn results/*.csv into the figures for the report.

Statistics follow the course's benchmarking rules: median with a
non-parametric (percentile bootstrap) 95% confidence interval, harmonic
mean for rates, no outlier removal.

    python3 scripts/plot_results.py results/strong.csv --kind strong
    python3 scripts/plot_results.py results/shm_ablation.csv --kind ablation
    python3 scripts/plot_results.py results/csweep.csv --kind csweep
    python3 scripts/plot_results.py results/shapes.csv --kind shapes
"""
import argparse, csv, math, os, random, statistics as st
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False


def boot_ci(xs, n=2000, lo=2.5, hi=97.5):
    """Percentile bootstrap CI of the median -- no normality assumed."""
    if len(xs) < 2:
        return (xs[0], xs[0]) if xs else (0.0, 0.0)
    meds = []
    for _ in range(n):
        meds.append(st.median(random.choices(xs, k=len(xs))))
    meds.sort()
    return meds[int(lo / 100 * len(meds))], meds[int(hi / 100 * len(meds)) - 1]


def harmonic(xs):
    xs = [x for x in xs if x > 0]
    return len(xs) / sum(1.0 / x for x in xs) if xs else 0.0


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def group(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def summarize(rs):
    t = [float(r["time_s"]) for r in rs]
    gf = [float(r["gflops"]) for r in rs]
    med = st.median(t)
    lo, hi = boot_ci(t)
    return dict(n=len(t), median=med, lo=lo, hi=hi,
                gflops=harmonic(gf),
                wire=st.median([float(r["wire_gb"]) for r in rs]),
                comm=st.median([float(r["t_comm"]) for r in rs]),
                comp=st.median([float(r["t_comp"]) for r in rs]))


def save(fig, name):
    os.makedirs("plots", exist_ok=True)
    p = os.path.join("plots", name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("wrote", p)


def kind_strong(rows):
    g = group(rows, ["P"])
    Ps = sorted(int(k[0]) for k in g)
    med = {p: summarize(g[(str(p),)]) for p in Ps}
    print(f"{'P':>6} {'median[s]':>11} {'95% CI':>22} {'Gflop/s':>10} {'speedup':>9} {'eff':>7}")
    for p in Ps:
        m = med[p]
        sp = med[Ps[0]]["median"] / m["median"]
        print(f"{p:>6} {m['median']:>11.5f} [{m['lo']:.5f},{m['hi']:.5f}] "
              f"{m['gflops']:>10.1f} {sp:>9.2f} {sp/(p/Ps[0]):>7.2f}")
    if HAVE_PLT:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].errorbar(Ps, [med[p]["median"] for p in Ps],
                       yerr=[[med[p]["median"] - med[p]["lo"] for p in Ps],
                             [med[p]["hi"] - med[p]["median"] for p in Ps]],
                       marker="o", capsize=3)
        ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
        ax[0].set_xlabel("processes P"); ax[0].set_ylabel("time [s]")
        ax[0].set_title("Strong scaling (median, 95% CI)")
        sp = [med[Ps[0]]["median"] / med[p]["median"] for p in Ps]
        ax[1].plot(Ps, sp, marker="o", label="measured")
        ax[1].plot(Ps, [p / Ps[0] for p in Ps], "--", label="ideal")
        ax[1].set_xscale("log", base=2); ax[1].set_yscale("log", base=2)
        ax[1].set_xlabel("processes P"); ax[1].set_ylabel("speedup")
        ax[1].legend(); ax[1].set_title("Speedup")
        save(fig, "strong_scaling.png")


def kind_ablation(rows):
    g = group(rows, ["bcast", "lookahead", "gridmap"])
    keys = sorted(g)
    print(f"{'policy':>22} {'median[s]':>11} {'95% CI':>22} {'comm[s]':>9} {'offnode GB':>11}")
    base = None
    for k in keys:
        m = summarize(g[k])
        lbl = f"{k[0]}{'+la' if k[1]=='1' else ''}/{k[2]}"
        if base is None:
            base = m["median"]
        print(f"{lbl:>22} {m['median']:>11.5f} [{m['lo']:.5f},{m['hi']:.5f}] "
              f"{m['comm']:>9.4f} {m['wire']:>11.3f}   ({base/m['median']:.2f}x)")
    if HAVE_PLT:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        lbls = [f"{k[0]}{'+la' if k[1]=='1' else ''}\n{k[2]}" for k in keys]
        ms = [summarize(g[k]) for k in keys]
        ax[0].bar(lbls, [m["median"] for m in ms],
                  yerr=[[m["median"] - m["lo"] for m in ms],
                        [m["hi"] - m["median"] for m in ms]], capsize=3)
        ax[0].set_ylabel("time [s]"); ax[0].set_title("Broadcast policy")
        ax[1].bar(lbls, [m["wire"] for m in ms], color="tab:orange")
        ax[1].set_ylabel("off-node panel traffic [GB]")
        ax[1].set_title("Bytes crossing a node boundary")
        save(fig, "bcast_ablation.png")


def kind_csweep(rows):
    g = group(rows, ["c"])
    cs = sorted(int(k[0]) for k in g)
    print(f"{'c':>4} {'median[s]':>11} {'95% CI':>22} {'comm[s]':>9}")
    for c in cs:
        m = summarize(g[(str(c),)])
        print(f"{c:>4} {m['median']:>11.5f} [{m['lo']:.5f},{m['hi']:.5f}] {m['comm']:>9.4f}")
    if HAVE_PLT:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ms = [summarize(g[(str(c),)]) for c in cs]
        ax.errorbar(cs, [m["median"] for m in ms],
                    yerr=[[m["median"] - m["lo"] for m in ms],
                          [m["hi"] - m["median"] for m in ms]],
                    marker="o", capsize=3)
        P = int(rows[0]["P"])
        ax.axvline(P ** (1 / 3), ls="--", color="grey", label="c = P^(1/3)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("replication depth c"); ax.set_ylabel("time [s]")
        ax.legend(); ax.set_title("2.5D replication sweep")
        save(fig, "c_sweep.png")


def kind_shapes(rows):
    g = group(rows, ["tag"])
    shapes = sorted({k[0].split("_", 1)[1] for k in g})
    print(f"{'shape':>14} {'fixed[s]':>10} {'planner[s]':>11} {'speedup':>8}  planner grid")
    for s in shapes:
        try:
            f = summarize(g[(f"fixed_{s}",)])
            p = summarize(g[(f"plan_{s}",)])
        except KeyError:
            continue
        r = g[(f"plan_{s}",)][0]
        print(f"{s:>14} {f['median']:>10.4f} {p['median']:>11.4f} "
              f"{f['median']/p['median']:>8.2f}  {r['Pr']}x{r['Pc']}x{r['c']} b={r['b']}")
    if HAVE_PLT:
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = range(len(shapes))
        fv = [summarize(g[(f"fixed_{s}",)])["median"] for s in shapes]
        pv = [summarize(g[(f"plan_{s}",)])["median"] for s in shapes]
        ax.bar([x - 0.2 for x in xs], fv, 0.4, label="fixed 8x8 grid")
        ax.bar([x + 0.2 for x in xs], pv, 0.4, label="cost-model planner")
        ax.set_xticks(list(xs)); ax.set_xticklabels(shapes, rotation=20)
        ax.set_ylabel("time [s]"); ax.legend()
        ax.set_title("Planner vs fixed grid across matrix shapes")
        save(fig, "planner_shapes.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--kind", required=True,
                    choices=["strong", "ablation", "csweep", "shapes"])
    a = ap.parse_args()
    rows = load(a.csv)
    if not rows:
        raise SystemExit("no rows in " + a.csv)
    {"strong": kind_strong, "ablation": kind_ablation,
     "csweep": kind_csweep, "shapes": kind_shapes}[a.kind](rows)
    if not HAVE_PLT:
        print("\n(matplotlib not installed -- numbers only)")
