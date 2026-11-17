"""
DeepMark Benchmark Runner.
Executes the authentic 5-Pillar Black-Box evaluation pipeline on real datasets.
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.benchmarks.evaluate_models import run_benchmark

if __name__ == "__main__":
    run_benchmark()
