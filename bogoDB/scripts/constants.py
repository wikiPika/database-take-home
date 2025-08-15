#!/usr/bin/env python3
"""
Constants and configuration parameters for the BogoDB system.
"""

import os

# Get project root directory (one level up from scripts directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Graph parameters
NUM_NODES = 500
MAX_EDGES_PER_NODE = 3
MAX_TOTAL_EDGES = 2 * NUM_NODES

# Query parameters
NUM_QUERIES = 200
LAMBDA_PARAM = 0.1
MAX_DEPTH = 10000

# Random walk parameters
NUM_WALKS_PER_QUERY = 10

# Multiprocessing parameters
# Disable in sandboxed environments to avoid signaling restrictions
USE_MULTIPROCESSING = False
NUM_PROCESSES = 0  # 0 means use all available processors

# File paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INITIAL_GRAPH_FILE = os.path.join(DATA_DIR, "initial_graph.json")
QUERIES_FILE = os.path.join(DATA_DIR, "queries.json")
INITIAL_RESULTS_FILE = os.path.join(DATA_DIR, "initial_results.json")
EVALUATION_RESULTS_FILE = os.path.join(DATA_DIR, "evaluation_results.json")

# Random seed for reproducibility
RANDOM_SEED = 42
