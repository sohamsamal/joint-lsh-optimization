# Joint Optimization of Graph Partitions and Classifiers for Fast ANNS (Neural LSH-style)

This repo contains code and experiments for a **joint optimization** approach to learned indexing for Approximate Nearest Neighbor Search (ANNS).  
The method alternates between:

1. **Balanced graph partition refinement** on a k-NN graph (to preserve locality)
2. **Routing classifier training** to predict partitions (to make routing learnable)

The main artifact here is the experiment notebook plus an extracted Python script.

## Repo layout

- `notebooks/Joint_LSH_Optimization.ipynb` — primary experiment notebook (recommended entry point)
- `src/notebook_extracted.py` — code cells exported from the notebook (reference / starting point)
- `data/` — place datasets here (not included)
- `results/` — plots / logs / exported metrics

## Quick start

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Open and run the notebook

```bash
jupyter lab
# open notebooks/Joint_LSH_Optimization.ipynb
```

## Data

The notebook is designed to run on common ANNS benchmarks (e.g., **SIFT1M**) or any embedding matrix `X ∈ R^(n×d)` plus queries and ground truth neighbors.

Place files under `data/` and update paths inside the notebook accordingly.

## Notes on graph partitioning

If your workflow uses **KaHIP** (or a similar balanced partitioner), install it separately and make sure the executable / bindings are available on your PATH.
If the notebook uses a different partitioner (or a fallback), it will run with the Python dependencies above.

## Repro tips

- Set random seeds for numpy/torch in the first cells.
- Log `recall@k` vs candidate set sizes for a fair comparison to Neural LSH-style baselines.
- Keep candidate set metrics consistent across runs.

## License

MIT (see `LICENSE`).
