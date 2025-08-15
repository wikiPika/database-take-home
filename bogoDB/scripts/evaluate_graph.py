#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict

# Ensure scripts directory is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from scripts.random_walk import BogoDB, run_queries
from scripts.constants import *


def validate_graph(
    graph: Dict[str, Dict[str, float]],
    num_nodes: int,
    max_total_edges: int,
    max_edges_per_node: int,
) -> Tuple[bool, str]:
    """
    Validate that the graph meets all constraints.

    Args:
        graph: The graph to validate
        num_nodes: Number of nodes expected in the graph
        max_total_edges: Maximum total edges allowed
        max_edges_per_node: Maximum edges per node allowed

    Returns:
        Tuple of (is_valid, error_message)
    """
    print("\n" + "=" * 60)
    print("VALIDATING GRAPH CONSTRAINTS")
    print("=" * 60)

    # Check number of nodes
    if len(graph) != num_nodes:
        return False, f"Graph has {len(graph)} nodes, expected {num_nodes}"

    # Check total edges
    total_edges = sum(len(edges) for edges in graph.values())
    if total_edges > max_total_edges:
        return (
            False,
            f"Graph has {total_edges} edges, exceeding limit of {max_total_edges}",
        )

    # Check edges per node
    for node, edges in graph.items():
        if len(edges) > max_edges_per_node:
            return (
                False,
                f"Node {node} has {len(edges)} edges, exceeding limit of {max_edges_per_node}",
            )

    # Check edge weight range (0-10)
    for node, edges in graph.items():
        for target, weight in edges.items():
            if weight < 0 or weight > 10:
                return (
                    False,
                    f"Edge from {node} to {target} has invalid weight {weight}",
                )

    # All checks passed
    print(f"✅ Graph has {total_edges} edges (max: {max_total_edges})")
    print(
        f"✅ Maximum edges per node: {max(len(edges) for edges in graph.values())} (max: {max_edges_per_node})"
    )
    print(f"✅ All edge weights are within valid range (0-10)")
    print("✅ All graph constraints satisfied")

    return True, "Graph is valid"


def compare_results(
    initial_results: Dict[str, Any], optimized_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare initial and optimized query results.

    Args:
        initial_results: Results from initial graph
        optimized_results: Results from optimized graph

    Returns:
        Dictionary with comparison metrics
    """
    print("\n" + "=" * 60)
    print("COMPARING QUERY PERFORMANCE")
    print("=" * 60)

    # Extract detailed results
    initial_details = initial_results.get("detailed_results", [])
    optimized_details = optimized_results.get("detailed_results", [])

    # Analyze success rates
    initial_success = sum(
        1
        for r in initial_details
        if r.get("success", False) or r.get("is_success", False)
    )
    optimized_success = sum(
        1
        for r in optimized_details
        if r.get("success", False) or r.get("is_success", False)
    )

    initial_total = len(initial_details)
    optimized_total = len(optimized_details)

    initial_rate = initial_success / initial_total * 100 if initial_total > 0 else 0
    optimized_rate = (
        optimized_success / optimized_total * 100 if optimized_total > 0 else 0
    )

    success_change = optimized_rate - initial_rate

    # Extract successful path lengths
    initial_lengths = []
    optimized_lengths = []

    for result in initial_details:
        if result.get("success", False) or result.get("is_success", False):
            path_length = result.get("median_path_length", float("inf"))
            if path_length != float("inf"):
                initial_lengths.append(path_length)

    for result in optimized_details:
        if result.get("success", False) or result.get("is_success", False):
            path_length = result.get("median_path_length", float("inf"))
            if path_length != float("inf"):
                optimized_lengths.append(path_length)

    # Calculate medians
    initial_median = (
        statistics.median(initial_lengths) if initial_lengths else float("inf")
    )
    optimized_median = (
        statistics.median(optimized_lengths) if optimized_lengths else float("inf")
    )

    # Calculate improvement
    if initial_median == float("inf"):
        path_improvement_pct = 0
    elif optimized_median == float("inf"):
        path_improvement_pct = -100
    else:
        path_improvement_pct = (
            (initial_median - optimized_median) / initial_median * 100
        )

    # Calculate combined score
    if optimized_rate == 0:
        # If nothing succeeds, score is 0
        combined_score = 0
    elif initial_median == float("inf") or optimized_median == float("inf"):
        # If we can't compute improvement, use only success rate
        combined_score = optimized_rate
    else:
        # Calculate a multiplier based on path length improvement
        path_multiplier = np.log1p(initial_median / optimized_median)
        combined_score = optimized_rate * (1 + path_multiplier)

    # Print results
    print(f"\nSUCCESS RATE:")
    print(f"  Initial:   {initial_rate:.1f}% ({initial_success}/{initial_total})")
    print(f"  Optimized: {optimized_rate:.1f}% ({optimized_success}/{optimized_total})")

    if success_change != 0:
        print(
            f"  {'✅ Improvement' if success_change > 0 else '❌ Regression'}: {abs(success_change):.1f}%"
        )

    print(f"\nPATH LENGTHS (successful queries only):")
    print(
        f"  Initial:   {initial_median if initial_median != float('inf') else 'inf'} ({len(initial_lengths)}/{initial_total} queries)"
    )
    print(
        f"  Optimized: {optimized_median if optimized_median != float('inf') else 'inf'} ({len(optimized_lengths)}/{optimized_total} queries)"
    )

    if optimized_median != float("inf") and initial_median != float("inf"):
        print(
            f"  {'✅ Improvement' if path_improvement_pct > 0 else '❌ Regression'}: {abs(path_improvement_pct):.1f}%"
        )

    print(f"\nCOMBINED SCORE (success rate × path efficiency):")
    print(f"  Score: {combined_score:.2f}")
    print(f"  Higher is better, rewards both success and shorter paths")

    # Create simple visualization
    visualize_results(
        initial_details, optimized_details, initial_lengths, optimized_lengths
    )

    # Return metrics
    return {
        "initial_success_rate": initial_rate / 100,
        "optimized_success_rate": optimized_rate / 100,
        "initial_median": initial_median,
        "optimized_median": optimized_median,
        "path_improvement_pct": path_improvement_pct,
        "combined_score": combined_score,
    }


def visualize_results(
    initial_details, optimized_details, initial_lengths, optimized_lengths
):
    """
    Create a simple visualization of evaluation results.
    """
    # Calculate success rates
    initial_success = sum(
        1
        for r in initial_details
        if r.get("success", False) or r.get("is_success", False)
    )
    optimized_success = sum(
        1
        for r in optimized_details
        if r.get("success", False) or r.get("is_success", False)
    )

    initial_rate = (
        initial_success / len(initial_details) * 100 if initial_details else 0
    )
    optimized_rate = (
        optimized_success / len(optimized_details) * 100 if optimized_details else 0
    )

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot success rates
    ax1.bar([0, 1], [initial_rate, optimized_rate], color=["blue", "orange"], width=0.6)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Initial", "Optimized"])
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Query Success Rate")

    # Add value labels
    for i, v in enumerate([initial_rate, optimized_rate]):
        ax1.text(i, v + 2, f"{v:.1f}%", ha="center")

    # Plot median path lengths
    if initial_lengths or optimized_lengths:
        labels = []
        values = []

        if initial_lengths:
            labels.append("Initial")
            values.append(statistics.median(initial_lengths))

        if optimized_lengths:
            labels.append("Optimized")
            values.append(statistics.median(optimized_lengths))

        ax2.bar(
            range(len(values)),
            values,
            color=["blue", "orange"][: len(values)],
            width=0.6,
        )
        ax2.set_xticks(range(len(values)))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Median Path Length")
        ax2.set_title("Path Length Comparison")

        # Add value labels
        for i, v in enumerate(values):
            ax2.text(i, v + max(values) / 20, f"{v:.1f}", ha="center")
    else:
        # No successful paths
        ax2.text(
            0.5,
            0.5,
            "No successful paths to compare",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Path Length Comparison")

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(project_root, "data", "evaluation_comparison.png"))
    print(f"\nEvaluation visualization saved to data/evaluation_comparison.png")


def evaluate_graphs():
    """
    Main function to run the evaluation process.
    """
    # Set paths
    data_dir = os.path.join(project_root, "data")
    candidate_dir = os.path.join(project_root, "candidate_submission")

    initial_graph_file = os.path.join(data_dir, "initial_graph.json")
    queries_file = os.path.join(data_dir, "queries.json")
    initial_results_file = os.path.join(data_dir, "initial_results.json")
    optimized_graph_file = os.path.join(candidate_dir, "optimized_graph.json")
    optimized_results_file = os.path.join(data_dir, "optimized_results.json")

    # Check file existence
    if not os.path.exists(initial_graph_file):
        print(f"Error: Initial graph file '{initial_graph_file}' not found")
        return

    if not os.path.exists(queries_file):
        print(f"Error: Queries file '{queries_file}' not found")
        return

    if not os.path.exists(initial_results_file):
        print(f"Error: Initial results file '{initial_results_file}' not found")
        return

    if not os.path.exists(optimized_graph_file):
        print(f"Error: Optimized graph file '{optimized_graph_file}' not found")
        return

    # Load files
    with open(initial_graph_file, "r") as f:
        initial_graph = json.load(f)

    with open(queries_file, "r") as f:
        queries = json.load(f)

    with open(initial_results_file, "r") as f:
        initial_results = json.load(f)

    with open(optimized_graph_file, "r") as f:
        optimized_graph = json.load(f)

    # Validate the optimized graph
    is_valid, message = validate_graph(
        optimized_graph, NUM_NODES, MAX_TOTAL_EDGES, MAX_EDGES_PER_NODE
    )

    if not is_valid:
        print(f"Error: {message}")
        return

    # Run queries on optimized graph
    print("\nRunning queries on the optimized graph...")
    bogodb = BogoDB(optimized_graph_file)
    optimized_results = run_queries(bogodb, queries)

    # Save optimized results
    with open(optimized_results_file, "w") as f:
        json.dump(optimized_results, f, indent=2)

    # Compare results
    compare_results(initial_results, optimized_results)


if __name__ == "__main__":
    evaluate_graphs()
