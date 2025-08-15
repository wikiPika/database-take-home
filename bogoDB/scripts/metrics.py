#!/usr/bin/env python3
import math


def combined_score(initial_median, optimized_success_rate, optimized_median):
    """
    Combined score consistent with evaluate_graph.py.

    - If optimized success rate is 0: score = 0
    - If any median is infinite or non-positive: score = optimized_success_rate
    - Else: score = optimized_success_rate * (1 + log1p(initial_median / optimized_median))

    Returns a float (not percentage); callers may multiply by 100 if desired.
    """
    if optimized_success_rate <= 0:
        return 0.0
    if (
        initial_median is None
        or optimized_median is None
        or math.isinf(initial_median)
        or math.isinf(optimized_median)
        or optimized_median <= 0
    ):
        return optimized_success_rate
    path_multiplier = math.log1p(initial_median / optimized_median)
    return optimized_success_rate * (1 + path_multiplier)

