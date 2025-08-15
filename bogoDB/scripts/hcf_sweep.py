#!/usr/bin/env python3
import os
import sys
import json
import math
import time
from collections import Counter

# Headless plotting safety
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from scripts.constants import (
    NUM_NODES,
    INITIAL_GRAPH_FILE,
    QUERIES_FILE,
)
from scripts.random_walk import BogoDB, run_queries
from scripts.metrics import combined_score
from candidate_submission.optimize_graph import build_hot_core_funnel_graph


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def evaluate_params(initial_median, queries, H, J, r_core, r_out):
    graph = build_hot_core_funnel_graph(
        NUM_NODES,
        hot_core_size=H,
        jump=J,
        p_core_ring=r_core,
        p_core_skip=1 - r_core,
        p_out_ring=r_out,
        p_out_funnel=1 - r_out,
    )
    db = BogoDB(graph)
    res = run_queries(db, queries)
    sr = res.get("success_rate", 0.0)
    med = res.get("median_path_length", float("inf"))
    score = combined_score(initial_median, sr, med)
    return sr, med, score


def coarse_grid():
    # Extremely small coarse ranges for quick demo runtime
    H_vals = [40]
    J_vals = [7]
    r_core_vals = [0.6]
    r_out_vals = [0.05]
    return H_vals, J_vals, r_core_vals, r_out_vals


def refine_grid(best):
    H, J, r_core, r_out = best
    # Local neighborhood sweep
    H_vals = sorted(set([max(16, H - 8), H, min(96, H + 8)]))
    J_vals = sorted(set([max(3, J - 2), J, J + 2]))
    r_core_vals = [max(0.3, round(r_core - 0.05, 2)), r_core, min(0.8, round(r_core + 0.05, 2))]
    r_out_vals = [max(0.01, round(r_out - 0.02, 3)), r_out, min(0.2, round(r_out + 0.02, 3))]
    return H_vals, J_vals, r_core_vals, r_out_vals


def main():
    if not os.path.exists(INITIAL_GRAPH_FILE) or not os.path.exists(QUERIES_FILE):
        print("Missing data; run generate_initial_data.py first.")
        return 1

    initial_graph = load_json(INITIAL_GRAPH_FILE)
    queries = load_json(QUERIES_FILE)

    # Baseline median (for scoring)
    base_db = BogoDB(initial_graph)
    baseline = run_queries(base_db, queries)
    base_median = baseline.get("median_path_length", float("inf"))

    tested = []
    best_tuple = None
    best_score = -1

    def sweep(H_vals, J_vals, r_core_vals, r_out_vals, tag="coarse"):
        nonlocal best_tuple, best_score
        print(f"\n[{tag}] Sweeping {len(H_vals)*len(J_vals)*len(r_core_vals)*len(r_out_vals)} combos...")
        for H in H_vals:
            for J in J_vals:
                for r_core in r_core_vals:
                    for r_out in r_out_vals:
                        start = time.time()
                        sr, med, score = evaluate_params(base_median, queries, H, J, r_core, r_out)
                        tested.append((H, J, r_core, r_out, sr, med, score))
                        if score > best_score:
                            best_score = score
                            best_tuple = (H, J, r_core, r_out)
                        print(
                            f"H={H:>3} J={J:>2} r_core={r_core:>4} r_out={r_out:>5} | "
                            f"SR={sr*100:5.1f}% Med={med:8.2f} Score={score*100:7.2f}",
                            end="\r",
                        )
        print("\nDone.")

    # Coarse sweep
    sweep(*coarse_grid(), tag="coarse")
    print(f"Best after coarse: {best_tuple} -> score={best_score*100:.2f}")

    # Two rounds of local refinement
    for round_id in range(1, 2):
        H_vals, J_vals, r_core_vals, r_out_vals = refine_grid(best_tuple)
        sweep(H_vals, J_vals, r_core_vals, r_out_vals, tag=f"refine{round_id}")
        print(f"Best after refine {round_id}: {best_tuple} -> score={best_score*100:.2f}")

    # Save a quick scatter of tested points colored by score
    try:
        import numpy as np
        from matplotlib.colors import Normalize
        scores = [s for *_, s in tested]
        norm = Normalize(vmin=min(scores), vmax=max(scores))
        plt.figure(figsize=(8, 6))
        xs = [H for H, *_ in tested]
        ys = [J for _, J, *_ in tested]
        cs = [norm(s) for *_, s in tested]
        plt.scatter(xs, ys, c=cs, cmap="viridis")
        plt.colorbar(label="Score")
        plt.xlabel("H (hot_core_size)")
        plt.ylabel("J (jump)")
        plt.title("HCF sweep: Score by (H, J)")
        out_png = os.path.join(PROJECT_ROOT, "data", "hcf_sweep_HJ.png")
        plt.tight_layout()
        plt.savefig(out_png)
        print(f"Saved plot {out_png}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    print("\nTOP 5 configs:")
    top5 = sorted(tested, key=lambda x: x[-1], reverse=True)[:5]
    for (H, J, r_core, r_out, sr, med, score) in top5:
        print(
            f"H={H} J={J} r_core={r_core:.2f} r_out={r_out:.3f} | "
            f"SR={sr*100:.1f}% Med={med:.2f} Score={score*100:.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
