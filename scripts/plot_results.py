#!/usr/bin/env python3
"""Turn results/*.csv into the figures for the report.

Statistics follow the course's benchmarking rules: median with a
non-parametric (percentile bootstrap) 95% confidence interval, harmonic
mean for rates, no outlier removal.

    python3 scripts/plot_results.py results/strong.csv --kind strong
    python3 scripts/plot_results.py results/shm_ablation.csv --kind ablation
    python3 scripts/plot_results.py results/csweep.csv --kind csweep
    python3 scripts/plot_results.py results/shapes.csv --kind shapes
    python3 scripts/plot_results.py results/kernel.csv  --kind kernel
    python3 scripts/plot_results.py results/engines.csv --kind engines
    python3 scripts/plot_results.py results/bsweep.csv  --kind bsweep
    python3 scripts/plot_results.py results/weak.csv    --kind weak
    python3 scripts/plot_results.py results/hybrid.csv  --kind hybrid
    python3 scripts/plot_results.py results/kernel.csv  --kind roofline
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


# The bootstrap resamples, so an unseeded run redraws slightly different
# whiskers every time and the figures are not bit-reproducible. The point of
# regenerating figures from the CSVs is that the same data gives the same
# picture, so fix the seed. Override with GEMM2D_PLOT_SEED to check that a
# conclusion is not an artefact of one particular resampling.
random.seed(int(os.environ.get("GEMM2D_PLOT_SEED", "20260907")))


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


def read_roofs(path="results/roofs.csv"):
    """Roofs measured by bench/roofline. Absent -> figures simply omit them."""
    try:
        return {r["metric"]: float(r["value"]) for r in load(path)}
    except (FileNotFoundError, KeyError, ValueError):
        return {}


def kind_roofline(rows):
    """Node-level roofline. Arithmetic intensity is COMPULSORY traffic --
    no hardware counters were available, so this is the algorithmic bound,
    not measured DRAM traffic. Say so in the caption."""
    roofs = read_roofs()
    bw = roofs.get("stream_triad")
    ceil = roofs.get("openblas_dgemm")
    if not bw:
        raise SystemExit("need results/roofs.csv -- run: make roofline BLAS=1 && "
                         "./bench/roofline > results/roofs.csv")
    M = N = 1024; K = 256
    flops = 2.0 * M * N * K
    comp_bytes = (M * K + K * N + 2.0 * M * N) * 8.0   # A + B + C read&write
    ai = flops / comp_bytes
    g = group(rows, ["kernel", "threads"])
    pts = {k[0]: summarize(g[k])["gflops"] for k in g if k[1] == "1"}
    peak = 2.8 * 8 * 2 * 2      # 2.8 GHz, 8 lanes, 2 flop/FMA, 2 FMA units
    print(f"arithmetic intensity (compulsory) = {ai:.1f} flop/byte")
    print(f"ridge point = {ceil/bw:.1f} flop/byte  -> "
          f"{'compute' if ai > ceil/bw else 'memory'}-bound")
    for k, v in sorted(pts.items(), key=lambda x: x[1]):
        print(f"  {k:>8} {v:>7.2f} Gflop/s   {v/ceil*100:>5.1f}% of OpenBLAS"
              f"  {v/peak*100:>5.1f}% of theoretical peak")
    if HAVE_PLT:
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        xs = [2.0 ** e for e in range(-2, 9)]
        ax.plot(xs, [min(bw * x, ceil) for x in xs], "k-", lw=2,
                label=f"achievable roof ({bw:.1f} GB/s, {ceil:.0f} Gflop/s)")
        ax.axhline(peak, ls=":", color="grey", label=f"theoretical peak {peak:.0f}")
        for k, v in pts.items():
            ax.plot(ai, v, "o", ms=8, label=f"{k} ({v:.1f})")
        ax.axvline(ceil / bw, ls="--", color="lightgrey", lw=1)
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xlabel("arithmetic intensity [flop/byte]")
        ax.set_ylabel("Gflop/s")
        ax.set_title("Node roofline, single core")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3, which="both")
        save(fig, "roofline.png")


def kind_kernel(rows):
    """E1 -- local kernel ablation. Rates, so harmonic mean; the guide is
    explicit that arithmetic means of rates are wrong."""
    g = group(rows, ["kernel", "threads"])
    keys = sorted(g, key=lambda k: (int(k[1]), k[0]))
    print(f"{'kernel':>10} {'threads':>8} {'Gflop/s':>10} {'median[s]':>11} {'95% CI':>22}")
    for k in keys:
        m = summarize(g[k])
        print(f"{k[0]:>10} {k[1]:>8} {m['gflops']:>10.2f} {m['median']:>11.5f} "
              f"[{m['lo']:.5f},{m['hi']:.5f}]")
    if HAVE_PLT:
        threads = sorted({int(k[1]) for k in keys})
        kernels = sorted({k[0] for k in keys})
        fig, ax = plt.subplots(figsize=(6, 4))
        w = 0.8 / len(threads)
        for i, t in enumerate(threads):
            vals = [summarize(g[(kn, str(t))])["gflops"] if (kn, str(t)) in g else 0
                    for kn in kernels]
            ax.bar([x + i * w - 0.4 + w / 2 for x in range(len(kernels))], vals, w,
                   label=f"{t} thread{'s' if t > 1 else ''}")
        ceil = read_roofs().get("openblas_dgemm")
        if ceil:
            ax.axhline(ceil, ls="--", color="crimson", lw=1.2)
            ax.text(len(kernels) - 0.5, ceil, f" OpenBLAS {ceil:.0f}",
                    va="bottom", ha="right", fontsize=8, color="crimson")
        ax.set_xticks(range(len(kernels))); ax.set_xticklabels(kernels)
        ax.set_ylabel("Gflop/s (harmonic mean)"); ax.legend(fontsize=8)
        ax.set_title("Local kernel ablation")
        save(fig, "kernel_ablation.png")


def kind_engines(rows):
    """E5 -- engine comparison at fixed P."""
    g = group(rows, ["engine"])
    keys = sorted(g)
    print(f"{'engine':>10} {'median[s]':>11} {'95% CI':>22} {'Gflop/s':>10} {'rank GB':>9}")
    for k in keys:
        m = summarize(g[k])
        rk = st.median([float(r["rank_gb"]) for r in g[k]])
        print(f"{k[0]:>10} {m['median']:>11.5f} [{m['lo']:.5f},{m['hi']:.5f}] "
              f"{m['gflops']:>10.2f} {rk:>9.4f}")
    if HAVE_PLT:
        ms = [summarize(g[k]) for k in keys]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].bar([k[0] for k in keys], [m["median"] for m in ms],
                  yerr=[[m["median"] - m["lo"] for m in ms],
                        [m["hi"] - m["median"] for m in ms]], capsize=3)
        ax[0].set_ylabel("time [s]"); ax[0].set_title("Engine comparison")
        ax[1].bar([k[0] for k in keys],
                  [st.median([float(r["rank_gb"]) for r in g[k]]) for k in keys],
                  color="tab:orange")
        ax[1].set_ylabel("panel bytes received per rank [GB]")
        ax[1].set_title("Communication volume")
        save(fig, "engine_comparison.png")


def kind_bsweep(rows):
    """E7 -- panel width. The model says the bandwidth term is independent of
    b and only the latency term scales as 1/b, so a flat curve on one node is
    a confirmation of the model, not a null result."""
    g = group(rows, ["b"])
    bs = sorted(int(k[0]) for k in g)
    print(f"{'b':>6} {'median[s]':>11} {'95% CI':>22} {'comm[s]':>9} {'comp[s]':>9}")
    for b in bs:
        m = summarize(g[(str(b),)])
        print(f"{b:>6} {m['median']:>11.5f} [{m['lo']:.5f},{m['hi']:.5f}] "
              f"{m['comm']:>9.4f} {m['comp']:>9.4f}")
    if HAVE_PLT:
        ms = [summarize(g[(str(b),)]) for b in bs]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.errorbar(bs, [m["median"] for m in ms],
                    yerr=[[m["median"] - m["lo"] for m in ms],
                          [m["hi"] - m["median"] for m in ms]],
                    marker="o", capsize=3, label="total")
        ax.plot(bs, [m["comm"] for m in ms], marker="s", ls="--", label="communication")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("panel width b"); ax.set_ylabel("time [s]")
        ax.legend(); ax.set_title("Panel width sweep")
        save(fig, "b_sweep.png")


def kind_weak(rows):
    """E4 -- weak scaling at constant MEMORY per process: n = n0*sqrt(P/P0).
    Work per rank therefore grows as sqrt(P); the expected curve rises."""
    g = group(rows, ["P"])
    Ps = sorted(int(k[0]) for k in g)
    base = summarize(g[(str(Ps[0]),)])["median"]
    print(f"{'P':>6} {'n':>7} {'median[s]':>11} {'95% CI':>22} {'Gflop/s':>10} "
          f"{'t/t0':>7} {'sqrt(P/P0)':>11}")
    for p in Ps:
        m = summarize(g[(str(p),)])
        n = int(g[(str(p),)][0]["M"])
        ideal = math.sqrt(p / Ps[0])
        print(f"{p:>6} {n:>7} {m['median']:>11.5f} [{m['lo']:.5f},{m['hi']:.5f}] "
              f"{m['gflops']:>10.1f} {m['median']/base:>7.2f} {ideal:>11.2f}")
    if HAVE_PLT:
        ms = [summarize(g[(str(p),)]) for p in Ps]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(Ps, [m["median"] / base for m in ms],
                    yerr=[[(m["median"] - m["lo"]) / base for m in ms],
                          [(m["hi"] - m["median"]) / base for m in ms]],
                    marker="o", capsize=3, label="measured")
        ax.plot(Ps, [math.sqrt(p / Ps[0]) for p in Ps], "--", color="grey",
                label=r"$\sqrt{P/P_0}$  (constant memory/rank)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("processes P"); ax.set_ylabel("time / time(P0)")
        ax.legend(); ax.set_title("Weak scaling, constant memory per rank")
        save(fig, "weak_scaling.png")


def kind_hybrid(rows):
    """E10 -- ranks x threads at a fixed core count."""
    g = group(rows, ["P", "threads"])
    keys = sorted(g, key=lambda k: -int(k[0]))
    print(f"{'ranks':>6} {'threads':>8} {'median[s]':>11} {'95% CI':>22} "
          f"{'Gflop/s':>10} {'rank GB':>9}")
    for k in keys:
        m = summarize(g[k])
        rk = st.median([float(r["rank_gb"]) for r in g[k]])
        print(f"{k[0]:>6} {k[1]:>8} {m['median']:>11.5f} "
              f"[{m['lo']:.5f},{m['hi']:.5f}] {m['gflops']:>10.1f} {rk:>9.3f}")
    if HAVE_PLT:
        lbls = [f"{k[0]}x{k[1]}" for k in keys]
        ms = [summarize(g[k]) for k in keys]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar(lbls, [m["median"] for m in ms],
                  yerr=[[m["median"] - m["lo"] for m in ms],
                        [m["hi"] - m["median"] for m in ms]], capsize=3)
        ax[0].set_ylabel("time [s]"); ax[0].set_xlabel("ranks x threads")
        ax[0].set_title("Hybrid mapping at fixed core count")
        ax[1].bar(lbls, [st.median([float(r["rank_gb"]) for r in g[k]]) for k in keys],
                  color="tab:orange")
        ax[1].set_ylabel("panel bytes per rank [GB]"); ax[1].set_xlabel("ranks x threads")
        ax[1].set_title("Traffic per rank")
        save(fig, "hybrid_mapping.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--kind", required=True,
                    choices=["strong", "ablation", "csweep", "shapes",
                             "kernel", "engines", "bsweep", "weak", "hybrid",
                             "roofline"])
    a = ap.parse_args()
    rows = load(a.csv)
    if not rows:
        raise SystemExit("no rows in " + a.csv)
    {"strong": kind_strong, "ablation": kind_ablation,
     "csweep": kind_csweep, "shapes": kind_shapes,
     "kernel": kind_kernel, "engines": kind_engines, "bsweep": kind_bsweep,
     "weak": kind_weak, "hybrid": kind_hybrid,
     "roofline": kind_roofline}[a.kind](rows)
    if not HAVE_PLT:
        print("\n(matplotlib not installed -- numbers only)")
