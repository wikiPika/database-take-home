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

# Ensure project root on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from scripts.constants import (
    NUM_NODES,
    MAX_TOTAL_EDGES,
    MAX_EDGES_PER_NODE,
    INITIAL_GRAPH_FILE,
    QUERIES_FILE,
)
from scripts.random_walk import BogoDB, run_queries
from scripts.evaluate_graph import validate_graph

# Import builders from candidate_submission
from candidate_submission.optimize_graph import (
    build_figure_eight_graph,
    build_figure_clover_graph,
    build_figure_clover_with_reentry_graph,
    build_chord_graph,
    build_de_bruijn_like_graph,
    build_express_ring_graph,
)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_baseline(initial_graph, queries):
    db = BogoDB(initial_graph)
    return run_queries(db, queries)


def short_metrics(results):
    sr = results.get("success_rate", 0.0) * 100
    med = results.get("median_path_length", float("inf"))
    return sr, med


def combined_score(initial_median, opt_success_rate, opt_median):
    if opt_success_rate <= 0:
        return 0.0
    if math.isinf(initial_median) or math.isinf(opt_median) or opt_median <= 0:
        return opt_success_rate * 100
    path_multiplier = math.log1p(initial_median / opt_median)
    return (opt_success_rate * 100) * (1 + path_multiplier)


def print_header(title):
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def print_table(rows, headers):
    # Determine column widths
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(r):
        return "  ".join(str(x).ljust(w) for x, w in zip(r, widths))

    print(fmt_row(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(fmt_row(r))


def main():
    # Load inputs
    if not os.path.exists(INITIAL_GRAPH_FILE) or not os.path.exists(QUERIES_FILE):
        print("Error: Missing initial graph or queries. Run generate_initial_data.py first.")
        return 1

    initial_graph = load_json(INITIAL_GRAPH_FILE)
    queries = load_json(QUERIES_FILE)
    qcounts = Counter(queries)

    # Baseline
    print_header("Baseline Evaluation (Initial Graph)")
    baseline = ensure_baseline(initial_graph, queries)
    base_sr, base_median = short_metrics(baseline)
    print(f"Success rate: {base_sr:.1f}%  |  Median path: {base_median if not math.isinf(base_median) else 'inf'}")

    # Define optimizers
    optimizers = [
        ("figure_eight", lambda: build_figure_eight_graph(NUM_NODES, qcounts)),
        ("clover", lambda: build_figure_clover_graph(NUM_NODES, qcounts)),
        (
            "clover_reentry",
            lambda: build_figure_clover_with_reentry_graph(NUM_NODES, qcounts),
        ),
        ("chord_step22", lambda: build_chord_graph(NUM_NODES, step=22)),
        ("de_bruijn_like", lambda: build_de_bruijn_like_graph(NUM_NODES)),
        ("express_ring", lambda: build_express_ring_graph(NUM_NODES)),
    ]

    # Evaluate each optimizer
    results_rows = []
    for name, build in optimizers:
        print_header(f"Evaluating: {name}")
        graph = build()

        # Validate constraints
        ok, msg = validate_graph(graph, NUM_NODES, MAX_TOTAL_EDGES, MAX_EDGES_PER_NODE)
        if not ok:
            print(f"❌ Invalid graph for {name}: {msg}")
            sr = 0.0
            med = float("inf")
            score = 0.0
        else:
            db = BogoDB(graph)
            start = time.time()
            res = run_queries(db, queries)
            dur = time.time() - start
            sr, med = short_metrics(res)
            score = combined_score(base_median, sr / 100.0, med)
            print(
                f"✓ Success rate: {sr:.1f}% | Median path: {med if not math.isinf(med) else 'inf'} | Time: {dur:.2f}s"
            )

        results_rows.append(
            [
                name,
                f"{sr:.1f}%",
                f"{med if not math.isinf(med) else 'inf'}",
                f"{score:.2f}",
            ]
        )

    # Summary table
    print_header("Summary (Higher score is better)")
    headers = ("Optimizer", "Success", "Median Path", "Score")
    # Sort by score desc
    sorted_rows = sorted(
        results_rows,
        key=lambda r: float(r[3]) if r[3] not in ("inf", "nan") else -1.0,
        reverse=True,
    )
    print_table(sorted_rows, headers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

