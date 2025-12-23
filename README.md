# Jointly Learning Graph Partitions and Classifiers for Fast Nearest Neighbor Search

This project studies how to improve data-dependent Approximate Nearest Neighbor Search (ANNS) by jointly optimizing a balanced k-NN graph partition and a routing classifier, instead of training the classifier to mimic a fixed partition (as in Neural LSH). For full research report, go to **FINAL_REPORT.pdf**

---

## Abstract

Graph-based ANNS methods such as Neural LSH proceed in two phases: (1) build a k-NN graph and compute a balanced graph partition, and (2) train a classifier that extends this partition to \(\mathbb{R}^d\).  
This two-stage pipeline can be misaligned: the partition is optimized without considering the classifier’s representational limits, while the classifier is forced to imitate a partition it did not help shape.

This project explores two attempts at joint optimization:
1. a relaxed, fully differentiable partitioner using soft assignments, and  
2. a structured, alternating optimization scheme using hard labels with classifier updates.

We report both negative and positive findings.

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
