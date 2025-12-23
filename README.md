# Jointly Learning Graph Partitions and Classifiers for Fast Nearest Neighbor Search

This repository contains the code and experiments for the project **Jointly Learning Graph Partitions and Classifiers for Fast Nearest Neighbor Search** (Columbia University).

It studies how to improve data-dependent Approximate Nearest Neighbor Search (ANNS) by **jointly optimizing a balanced k-NN graph partition and a routing classifier**, instead of training the classifier to mimic a fixed partition (as in Neural LSH).

---

## Abstract

Graph-based ANNS methods such as Neural LSH proceed in two phases: (1) build a k-NN graph and compute a balanced graph partition, and (2) train a classifier that extends this partition to \(\mathbb{R}^d\).  
This two-stage pipeline can be misaligned: the partition is optimized without considering the classifier’s representational limits, while the classifier is forced to imitate a partition it did not help shape.

This project explores two attempts at joint optimization:
1. a relaxed, fully differentiable partitioner using soft assignments, and  
2. a structured, alternating optimization scheme using hard labels with classifier updates.

We report both negative and positive findings.

---

## Key Idea

Let \(P = \{x_1,\dots,x_n\} \subset \mathbb{R}^d\) and let \(G=(V,E)\) be its (approximate) k-NN graph.

Neural LSH:
- computes a balanced partition \(b:V\to[m]\) using a graph partitioner (e.g., KaHIP),
- then trains a classifier \(f_\theta\) to predict \(b(i)\).

This project asks: **can the partition be refined using the classifier so that both co-adapt?**

---

## Approach 1: Continuous Relaxation (Soft Assignments) — Failure Mode

We tried to learn a single function

\[
g_\theta : \mathbb{R}^d \to \Delta^{m-1}
\]

that outputs soft bin assignments \(a_i = g_\theta(x_i)\).  
Loss terms included:

- **Soft cut loss**  
  \[
  \ell_{\text{cut}}(i,j)= 1 - \sum_{c=1}^m a_{i,c}a_{j,c}
  \]

- **Balance penalty**  
  \[
  \ell_{\text{bal}} = \frac{1}{m}\sum_{c=1}^m\left(s_c - \frac{1}{m}\right)^2,\quad s_c = \frac{1}{n}\sum_i a_{i,c}
  \]

- **Smoothness (Laplacian-style)**  
  \[
  \ell_{\text{smooth}} = \frac{1}{|E|}\sum_{(i,j)\in E}\|a_i-a_j\|_2^2
  \]

- **Entropy regularization** to avoid one-hot collapse.

**Observed outcome:** despite extensive tuning, optimization consistently collapsed into a single bin (degenerate partition), because cut + smoothness are minimized by assigning every point to the same label and gradients vanish once softmax saturates.

Takeaway: a naive differentiable relaxation is **intrinsically unstable** for this objective.

---

## Approach 2: Alternating Refinement (Hard Labels) — Working Method

We instead start from a balanced partition \(b^{(0)}\) (KaHIP on the k-NN graph) and alternate:

### (A) Classifier update
With \(b^{(t)}\) fixed, train \(f_{\theta}\) to maximize

\[
\sum_{i=1}^n \log p_\theta\bigl(b^{(t)}(i)\mid x_i\bigr).
\]

### (B) Partition refinement (local search)
With \(\theta\) fixed, propose moves for vertices using the classifier’s top prediction

\[
\hat b_i = \arg\max_c p_\theta(c\mid x_i),
\]

and accept a move \(i\to \hat b_i\) only if:
- balance constraints remain satisfied, and
- graph locality strictly improves (more within-block neighbor weight).

---

## Theory (What We Prove)

Define within-block cohesion:

\[
\Phi(b)=\sum_{(i,j)\in E} w_{ij}\,\mathbf{1}[b_i=b_j].
\]

- **Exact local gain for a 1-vertex move**  
  \[
  \Phi(b^{(i\to c)})-\Phi(b)=W_i(c;b)-W_i(b_i;b)
  \]
  where \(W_i(c;b)=\sum_{j:(i,j)\in E} w_{ij}\mathbf{1}[b_j=c]\).

- **Finite termination**: any procedure that applies only strictly improving feasible moves must stop in finitely many steps (finite \(\mathcal B\), strictly increasing \(\Phi\)).

- **Local optimality**: the terminal partition is a 1-move local optimum w.r.t. the considered move set.

We also define a joint potential:

\[
\Psi(b,\theta)=\Phi(b)+\lambda\sum_i \log p_\theta(b_i\mid x_i)
\]

and show monotonicity under idealized alternating updates.

---

## Experiments

### Dataset
- **SIFT1M** (1,000,000 points, 128-D)
- k-NN graph with **k = 10**
- number of bins **m = 64**

### Metric
We evaluate the standard **recall–candidate tradeoff** (k-NN accuracy at k=10 vs. number of distance computations), including:
- average candidates per query
- 0.95-quantile candidates per query (tail latency proxy)

### Result Summary
The alternating refinement method **matches or improves** Neural LSH across:
- higher recall for the same candidate budget,
- fewer candidates for the same recall,
- improved 0.95-quantile behavior (better worst-case routing).

See `new_lsh.png` for the recall–candidate curves.

---

## Repository Structure (suggested)

- `notebooks/`  
  - `Joint_LSH_Optimization.ipynb` (main experiments)
- `src/`  
  - graph construction, partition utilities, training code
- `assets/`  
  - figures (e.g., `new_lsh.png`)
- `README.md`  
  - this file

If your repo layout differs, adjust section paths accordingly.

---

## How to Run

> Update these commands to match your actual scripts/paths.

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run experiments**
   ```bash
   python -m src.run_experiment --dataset sift1m --k 10 --m 64
   ```

3. **Reproduce plots**
   ```bash
   python -m src.plot_curves --input results/ --output assets/new_lsh.png
   ```

---

## References

- Indyk & Motwani (1998), *Approximate nearest neighbors: towards removing the curse of dimensionality*  
- Andoni & Indyk (2008), *Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions*  
- Andoni et al. (2015), *Optimal data-dependent hashing for approximate near neighbors*  
- Dong et al. (2020), *Neural Locality-Sensitive Hashing*  
- Sanders & Schulz (2013), *KaHIP: Karlsruhe High Quality Partitioning*  
- Shi & Malik (2000), *Normalized Cuts and Image Segmentation*  
- von Luxburg (2007), *A Tutorial on Spectral Clustering*  
- Dempster et al. (1977), *Maximum likelihood from incomplete data via the EM algorithm*  
- Boyd & Vandenberghe (2004), *Convex Optimization*  
- Kernighan & Lin (1970), *An efficient heuristic procedure for partitioning graphs*

---

## Acknowledgments

GPT-5 assisted with summarization/polishing and proof thinking (as noted in the final report).
