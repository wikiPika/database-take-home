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


def build_figure_clover_graph(
    num_nodes: int,
    query_counts: Counter,
    seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """
    Construct a figure-clover graph with three loops attached to the top-1 hub.

    - Three cycles (A, B, C) share a single center node (top-1 by query count).
    - All nodes have outdegree 1, except the center which has outdegree 3.
    - All edge weights are 1.0.
    - Frequently queried nodes are split across the three circles and spaced within each.
    """
    random.seed(seed)

    all_nodes = list(range(num_nodes))
    if query_counts:
        max_freq = max(query_counts.values())
        candidates = [n for n, c in query_counts.items() if c == max_freq]
        center = min(candidates)
    else:
        center = 0

    remaining = [n for n in all_nodes if n != center]
    freq = {n: query_counts.get(n, 0) for n in remaining}

    high_nodes = [n for n in remaining if freq[n] > 0]
    high_nodes.sort(key=lambda n: (-freq[n], n))

    # Target sizes for three circles
    base = len(remaining) // 3
    sizes = [base, base, len(remaining) - 2 * base]

    sets = [set(), set(), set()]
    # Round-robin allocate high nodes to balance across sets
    idx = 0
    for n in high_nodes:
        # Find next set that still has capacity
        assigned = False
        for _ in range(3):
            si = (idx) % 3
            if len(sets[si]) < sizes[si]:
                sets[si].add(n)
                idx = (si + 1)
                assigned = True
                break
            idx += 1
        if not assigned:
            # Fallback: put in the smallest set
            si = min(range(3), key=lambda i: len(sets[i]))
            sets[si].add(n)

    # Fill remaining capacity randomly
    random.shuffle(remaining)
    for n in remaining:
        if any(n in s for s in sets):
            continue
        # choose the set with remaining capacity, else the smallest
        choices = [i for i in range(3) if len(sets[i]) < sizes[i]]
        if choices:
            si = random.choice(choices)
        else:
            si = min(range(3), key=lambda i: len(sets[i]))
        sets[si].add(n)

    circles = [list(s) for s in sets]

    def spaced_order(circle_nodes: List[int]) -> List[int]:
        N = len(circle_nodes)
        if N == 0:
            return []
        local_high = [n for n in high_nodes if n in circle_nodes]
        local_rest = [n for n in circle_nodes if n not in local_high]
        order = [None] * N
        if local_high:
            positions = [int((i + 0.5) * N / len(local_high)) % N for i in range(len(local_high))]
            used = set()
            fixed = []
            for p in positions:
                while p in used:
                    p = (p + 1) % N
                used.add(p)
                fixed.append(p)
            for pos, node in zip(fixed, local_high):
                order[pos] = node
        random.shuffle(local_rest)
        it = iter(local_rest)
        for i in range(N):
            if order[i] is None:
                order[i] = next(it)
        return order

    orders = [spaced_order(c) for c in circles]

    graph: Dict[str, Dict[str, float]] = {str(n): {} for n in all_nodes}

    # Build edges around each circle, last -> center
    for order in orders:
        for i in range(len(order)):
            src = order[i]
            if i + 1 < len(order):
                dst = order[i + 1]
            else:
                dst = center
            graph[str(src)][str(dst)] = 1.0

    # Center to the start of each circle if non-empty
    for order in orders:
        if order:
            graph[str(center)][str(order[0])] = 1.0

    return graph


def build_figure_clover_with_reentry_graph(
    num_nodes: int,
    query_counts: Counter,
    seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """
    Figure-clover with a mid-loop re-entry edge from each clove back to the hub.

    - Start with the three-loop clover.
    - For each loop, add one extra edge from its midpoint node back to the center (hub).
    - All weights remain 1.0.
    """
    # Build base clover
    base = build_figure_clover_graph(num_nodes, query_counts, seed=seed)

    # Reconstruct the same orders used by the base builder by re-running its assignment logic
    # Note: To keep in sync, we repeat the circle construction deterministically using the same seed.
    random.seed(seed)
    all_nodes = list(range(num_nodes))
    if query_counts:
        max_freq = max(query_counts.values())
        candidates = [n for n, c in query_counts.items() if c == max_freq]
        center = min(candidates)
    else:
        center = 0

    remaining = [n for n in all_nodes if n != center]
    freq = {n: query_counts.get(n, 0) for n in remaining}
    high_nodes = [n for n in remaining if freq[n] > 0]
    high_nodes.sort(key=lambda n: (-freq[n], n))

    base_size = len(remaining) // 3
    sizes = [base_size, base_size, len(remaining) - 2 * base_size]
    sets = [set(), set(), set()]
    idx = 0
    for n in high_nodes:
        assigned = False
        for _ in range(3):
            si = (idx) % 3
            if len(sets[si]) < sizes[si]:
                sets[si].add(n)
                idx = (si + 1)
                assigned = True
                break
            idx += 1
        if not assigned:
            si = min(range(3), key=lambda i: len(sets[i]))
            sets[si].add(n)

    random.shuffle(remaining)
    for n in remaining:
        if any(n in s for s in sets):
            continue
        choices = [i for i in range(3) if len(sets[i]) < sizes[i]]
        if choices:
            si = random.choice(choices)
        else:
            si = min(range(3), key=lambda i: len(sets[i]))
        sets[si].add(n)

    circles = [list(s) for s in sets]

    def spaced_order(circle_nodes: List[int]) -> List[int]:
        N = len(circle_nodes)
        if N == 0:
            return []
        local_high = [n for n in high_nodes if n in circle_nodes]
        local_rest = [n for n in circle_nodes if n not in local_high]
        order = [None] * N
        if local_high:
            positions = [int((i + 0.5) * N / len(local_high)) % N for i in range(len(local_high))]
            used = set()
            fixed = []
            for p in positions:
                while p in used:
                    p = (p + 1) % N
                used.add(p)
                fixed.append(p)
            for pos, node in zip(fixed, local_high):
                order[pos] = node
        random.shuffle(local_rest)
        it = iter(local_rest)
        for i in range(N):
            if order[i] is None:
                order[i] = next(it)
        return order

    orders = [spaced_order(c) for c in circles]

    # Add one re-entry edge per loop from its midpoint back to center
    for order in orders:
        if not order:
            continue
        mid = len(order) // 2
        src = order[mid]
        # Outgoing edge from midpoint to center (weight 1.0)
        base[str(src)][str(center)] = 1.0

    return base


def build_chord_graph(
    num_nodes: int,
    step: int = 22,
) -> Dict[str, Dict[str, float]]:
    """
    Build a Chord-style ring with a fixed skip edge per node.

    - Each node i points to (i+1) % N and (i+step) % N.
    - All weights are 1.0.
    - Results in exactly 2*N edges (here, 1000), max outdegree=2.
    """
    graph: Dict[str, Dict[str, float]] = {str(i): {} for i in range(num_nodes)}
    for i in range(num_nodes):
        succ = (i + 1) % num_nodes
        skip = (i + step) % num_nodes
        graph[str(i)][str(succ)] = 1.0
        graph[str(i)][str(skip)] = 1.0
    return graph


def build_de_bruijn_like_graph(
    num_nodes: int,
) -> Dict[str, Dict[str, float]]:
    """
    Build a de Bruijn-inspired graph on integers 0..N-1:
    For each node i, add edges to (2*i) % N and (2*i + 1) % N, weight 1.0.

    Notes:
    - Outdegree per node: 2; total edges: 2*N (≤ 1000 when N=500).
    - We interpret nodes as 0-based to match existing data (0..499).
    """
    graph: Dict[str, Dict[str, float]] = {str(i): {} for i in range(num_nodes)}
    for i in range(num_nodes):
        a = (2 * i) % num_nodes
        b = (2 * i + 1) % num_nodes
        graph[str(i)][str(a)] = 1.0
        graph[str(i)][str(b)] = 1.0
    return graph


def build_express_ring_graph(num_nodes: int) -> Dict[str, Dict[str, float]]:
    """
    Build a ring over 0..N-1 (i -> (i+1)%N) and add an "express" train:
    Using 1-based description: 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 1
    Mapped to 0-based indices: 0 -> 1 -> 3 -> 7 -> 15 -> 31 -> 63 -> 127 -> 255 -> 0
    All edge weights are 1.0.
    """
    graph: Dict[str, Dict[str, float]] = {str(i): {} for i in range(num_nodes)}

    # Ring edges
    for i in range(num_nodes):
        nxt = (i + 1) % num_nodes
        graph[str(i)][str(nxt)] = 1.0

    # Express train edges (0-based mapping of powers-of-two cycle)
    express = [0, 1, 3, 7, 15, 31, 63, 127, 255, 0]
    for s, t in zip(express, express[1:]):
        graph[str(s)][str(t)] = 1.0

    return graph


def build_de_brujin_enhanced(num_nodes: int) -> Dict[str, Dict[str, float]]:
    """
    Enhanced de Bruijn-inspired topology that favors reaching low IDs quickly.

    For each node i (0-based):
    - Edge A: to (2*i) % N  (doubling edge keeps expander-like reachability)
    - Edge B: to i//2 if i > 0 else 1  (halving edge funnels toward small IDs)

    All weights are 1.0. Total edges = 2*N, outdegree per node = 2.
    This design leverages the workload skew where lower IDs are queried more often.
    """
    graph: Dict[str, Dict[str, float]] = {str(i): {} for i in range(num_nodes)}
    for i in range(num_nodes):
        a = (2 * i) % num_nodes
        b = (i // 2) if i > 0 else 1  # avoid self-loop at 0 -> 0
        graph[str(i)][str(a)] = 1.0
        graph[str(i)][str(b)] = 1.0
    return graph


def build_hot_core_funnel_graph(
    num_nodes: int,
    hot_core_size: int = 40,
    jump: int = 7,
    p_core_ring: float = 0.6,
    p_core_skip: float = 0.4,
    p_out_ring: float = 0.05,
    p_out_funnel: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    Hot-Core Funnel (HCF) topology as specified:

    - Nodes 0..(H-1) form the hot core.
    - All nodes have ring edge i -> (i+1)%N.
      * Core ring weight = p_core_ring
      * Outside ring weight = p_out_ring
    - Second edge per node:
      * Outside nodes: funnel to node 0 with weight p_out_funnel
      * Core nodes: skip to ((i+jump) % hot_core_size) with weight p_core_skip

    Weights are chosen to sum to 1 per node.
    """
    H = hot_core_size
    graph: Dict[str, Dict[str, float]] = {str(i): {} for i in range(num_nodes)}

    for i in range(num_nodes):
        nxt = (i + 1) % num_nodes
        if i < H:
            # Core node
            graph[str(i)][str(nxt)] = float(p_core_ring)
            skip = (i + jump) % H
            graph[str(i)][str(skip)] = float(p_core_skip)
        else:
            # Outside node
            graph[str(i)][str(nxt)] = float(p_out_ring)
            graph[str(i)]["0"] = float(p_out_funnel)

    return graph

def build_modified_hot_core(num_nodes):
    return build_hot_core_funnel_graph(
        num_nodes=num_nodes,
        hot_core_size=32,
        p_core_ring=0.65,
        p_core_skip=0.35,
        p_out_ring=0.05,
        p_out_funnel=0.95,
    )


def optimize_graph(
    initial_graph,
    queries: List[int],
    num_nodes=NUM_NODES,
    max_total_edges=int(MAX_TOTAL_EDGES),
    max_edges_per_node=MAX_EDGES_PER_NODE,
):
    """Build the figure-clover topology (three loops) for testing."""
    print("Starting Hot-Core Funnel graph construction...")

    optimized_graph = build_hot_core_funnel_graph(num_nodes)

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
