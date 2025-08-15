#!/usr/bin/env python3
import json
import os
import sys
import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

# Add project root to path to import scripts
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.append(project_dir)

# Import constants
from scripts.constants import (
    NUM_NODES,
    MAX_EDGES_PER_NODE,
    MAX_TOTAL_EDGES,
    QUERIES_FILE,
    RANDOM_SEED,
)


def load_graph(graph_file):
    """Load graph from a JSON file."""
    with open(graph_file, "r") as f:
        return json.load(f)


def load_results(results_file):
    """Load query results from a JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def save_graph(graph, output_file):
    """Save graph to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(graph, f, indent=2)


def verify_constraints(graph, max_edges_per_node, max_total_edges):
    """Verify that the graph meets all constraints."""
    # Check total edges
    total_edges = sum(len(edges) for edges in graph.values())
    if total_edges > max_total_edges:
        print(
            f"WARNING: Graph has {total_edges} edges, exceeding limit of {max_total_edges}"
        )
        return False

    # Check max edges per node
    max_node_edges = max(len(edges) for edges in graph.values())
    if max_node_edges > max_edges_per_node:
        print(
            f"WARNING: A node has {max_node_edges} edges, exceeding limit of {max_edges_per_node}"
        )
        return False

    # Check all nodes are present
    if len(graph) != NUM_NODES:
        print(f"WARNING: Graph has {len(graph)} nodes, should have {NUM_NODES}")
        return False

    # Check edge weights are valid (between 0 and 10)
    for node, edges in graph.items():
        for target, weight in edges.items():
            if weight <= 0 or weight > 10:
                print(f"WARNING: Edge {node} -> {target} has invalid weight {weight}")
                return False

    return True


def build_figure_eight_graph(
    num_nodes: int,
    query_counts: Counter,
    seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """Construct a figure-eight graph where the top-1 target is the center.

    - Two cycles (A and B) share a single center node (top-1 most queried).
    - All nodes have outdegree 1, except the center which has outdegree 2.
    - All edge weights are 1.0.
    - Nodes (especially frequently queried ones) are spaced across and within circles.
    """
    random.seed(seed)

    # Determine the center node (top-1 by query count; tie breaks by smallest ID)
    all_nodes = list(range(num_nodes))
    if query_counts:
        max_freq = max(query_counts.values())
        candidates = [n for n, c in query_counts.items() if c == max_freq]
        center = min(candidates)
    else:
        center = 0

    # Remaining nodes exclude the center
    remaining = [n for n in all_nodes if n != center]

    # Frequency map for remaining nodes
    freq = {n: query_counts.get(n, 0) for n in remaining}

    # Identify nodes that were ever queried (besides center)
    high_nodes = [n for n in remaining if freq[n] > 0]
    # Sort by frequency desc, then by node id for determinism
    high_nodes.sort(key=lambda n: (-freq[n], n))

    # Base random distribution across circles
    random.shuffle(remaining)
    size_a = len(remaining) // 2
    size_b = len(remaining) - size_a

    # Build circle assignments ensuring high_nodes are split evenly
    circle_a_set, circle_b_set = set(), set()
    toggle = True
    for n in high_nodes:
        if toggle and len(circle_a_set) < size_a:
            circle_a_set.add(n)
        elif not toggle and len(circle_b_set) < size_b:
            circle_b_set.add(n)
        elif len(circle_a_set) < size_a:
            circle_a_set.add(n)
        else:
            circle_b_set.add(n)
        toggle = not toggle

    # Fill remaining slots randomly while preserving sizes
    for n in remaining:
        if n in circle_a_set or n in circle_b_set:
            continue
        if len(circle_a_set) < size_a:
            circle_a_set.add(n)
        else:
            circle_b_set.add(n)

    circle_a = list(circle_a_set)
    circle_b = list(circle_b_set)

    # Equal spacing of high-frequency nodes within each circle
    def spaced_order(circle_nodes: List[int]) -> List[int]:
        N = len(circle_nodes)
        if N == 0:
            return []
        # Split into high vs others for this circle
        local_high = [n for n in high_nodes if n in circle_nodes]
        local_rest = [n for n in circle_nodes if n not in local_high]

        order = [None] * N
        if local_high:
            # Place local_high at approximately equal intervals
            positions = [int((i + 0.5) * N / len(local_high)) % N for i in range(len(local_high))]
            # Sort to make positions unique and ascending; in rare collisions, adjust
            used = set()
            fixed_positions = []
            for p in positions:
                while p in used:
                    p = (p + 1) % N
                used.add(p)
                fixed_positions.append(p)

            for pos, node in zip(fixed_positions, local_high):
                order[pos] = node

        random.shuffle(local_rest)
        it = iter(local_rest)
        for i in range(N):
            if order[i] is None:
                order[i] = next(it)
        return order

    order_a = spaced_order(circle_a)
    order_b = spaced_order(circle_b)

    # Build adjacency: all weights 1.0
    graph: Dict[str, Dict[str, float]] = {str(n): {} for n in all_nodes}

    # Circle A edges -> next, last -> center
    for i in range(len(order_a)):
        src = order_a[i]
        if i + 1 < len(order_a):
            dst = order_a[i + 1]
        else:
            dst = center
        graph[str(src)][str(dst)] = 1.0

    # Circle B edges -> next, last -> center
    for i in range(len(order_b)):
        src = order_b[i]
        if i + 1 < len(order_b):
            dst = order_b[i + 1]
        else:
            dst = center
        graph[str(src)][str(dst)] = 1.0

    # Center connects out to the start of both circles (if present)
    if order_a:
        graph[str(center)][str(order_a[0])] = 1.0
    if order_b:
        graph[str(center)][str(order_b[0])] = 1.0

    return graph


def optimize_graph(
    initial_graph,
    queries: List[int],
    num_nodes=NUM_NODES,
    max_total_edges=int(MAX_TOTAL_EDGES),
    max_edges_per_node=MAX_EDGES_PER_NODE,
):
    """Build the figure-eight topology based on query frequencies."""
    print("Starting figure-eight graph construction...")

    # Compute query frequencies
    counts = Counter(queries)

    optimized_graph = build_figure_eight_graph(num_nodes, counts, seed=RANDOM_SEED)

    # Verify constraints
    if not verify_constraints(optimized_graph, max_edges_per_node, max_total_edges):
        print("WARNING: Your optimized graph does not meet the constraints!")
        print("The evaluation script will reject it. Please fix the issues.")

    return optimized_graph


if __name__ == "__main__":
    # Get file paths
    initial_graph_file = os.path.join(project_dir, "data", "initial_graph.json")
    queries_file = os.path.join(project_dir, "data", "queries.json")
    output_file = os.path.join(
        project_dir, "candidate_submission", "optimized_graph.json"
    )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Loading initial graph from {initial_graph_file}")
    initial_graph = load_graph(initial_graph_file)

    print(f"Loading queries from {queries_file}")
    with open(queries_file, "r") as f:
        queries = json.load(f)

    print("Building figure-eight optimized graph...")
    optimized_graph = optimize_graph(initial_graph, queries)

    print(f"Saving optimized graph to {output_file}")
    save_graph(optimized_graph, output_file)

    print("Done! Optimized graph has been saved.")
