#!/usr/bin/env python3
"""Minimal entrypoint.

This script exists so the repo has a clear 'python -m' style entrypoint.
Most experiments live in the notebook.

Example:
  python src/run.py --help
"""

import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--note", default="Run notebooks/Joint_LSH_Optimization.ipynb for full experiments.")
    args = p.parse_args()
    print(args.note)

if __name__ == "__main__":
    main()
