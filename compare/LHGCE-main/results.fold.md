The results below are executed for only # one fold to examine the best approach available

# GENERAL
| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=6500, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 39.21% | 83.70% | 46.21% | 97.29% | 46.13% | 98.61% | 46.29% |

---
# variable: n
| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=16, negative_split='all', same=False, thr_percent=2.5 | 33.63% | 85.49% | 36.70% | 96.81% | 36.68% | 98.36% | 36.73% | 
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=32, negative_split='all', same=False, thr_percent=2.5 | 37.60% | 84.56% | 40.44% | 97.51% | 33.42% | 99.17% | 51.20% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 48.95% | 83.79% | 54.24% | 98.29% | 40.13% | 99.79% | 83.69% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=96, negative_split='all', same=False, thr_percent=2.5 | 42.31% | 84.19% | 49.89% | 97.47% | 49.82% | 98.70% | 49.95% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=128, negative_split='all', same=False, thr_percent=2.5 | 54.43% | 84.43% | 61.87% | 98.55% | 46.42% | 99.90% | 92.73% |

 [0.54438477 0.84432786 0.61878053 0.98558113 0.46429542 0.99905929
 0.92733186]
---

# variable: lr
| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 
| aggregator='mul', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=1, lr_encoder=0.01, lr_linear=0.01, n=64, negative_split='all', same=False, thr_percent=2.5  | 48.82%  | 81.89% | 60.35% | 98.56% | 43.30% | 99.99% | 99.56% |
| aggregator='mul', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=1, lr_encoder=0.005, lr_linear=0.005, n=64, negative_split='all', same=False, thr_percent=2.5 | 38.34% | 81.89% | 46.94% | 97.33% | 46.80% | 98.63% | 47.07% |
| aggregator='mul', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=1, lr_encoder=0.0001, lr_linear=0.0001, n=64, negative_split='all', same=False, thr_percent=2.5 | 55.93% | 84.54%  | 64.73% | 98.67% | 48.33% | 99.97% | 98.01% |


---

#variable: epochs 
| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 

---

# variable: aggregator 
| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 
| aggregator='sum', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 35.84% | 84.34% | 40.54% | 97% | 40.48% | 98.46% | 40.60% |
| aggregator='mul', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 60.92% | 84.88% | 71.01% | 98.86% | 55.30% | 99.98% | 99.17% | 
| aggregator='mean', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 38.10% | 85.22% | 40.99% | 97.59% | 33.20% | 99.25% | 53.56% | 

---

# variable: l 
| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 

---

# variable: l and epochs 
| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=4000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 37.77% | 84.57% | 43.44% | 97.15% | 43.38% | 98.54% | 43.49% | 
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=5500, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 47.75% | 84.36% | 52.12% | 98.22% | 38.36% | 99.77% | 81.25% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=6000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 48.32% | 84.21% | 52.67% | 98.20% | 39.53% | 99.72% | 78.91% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=6500, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 38.91% | 83.80% | 45.81% | 97.27% | 45.69% | 98.60% | 45.93% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 48.77% | 84.03% | 53.86% | 98.30% | 39.28% | 99.82% | 85.62% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=7500, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 39% | 83.61% | 45.96% | 97.27% | 45.91% | 98.60% | 46.02% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=8000, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 39.05% | 83.57% | 46.05% | 97.28% | 45.96% | 98.61% | 46.14% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=8500, fold=2, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 50.45% | 83.65% | 57.17% | 98.42% | 41.75% | 99.88% | 90.63% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=5500, fold=2, folds=5, l=2, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 36.31% | 84.44% | 41.42% | 97.05% | 41.35% | 98.49% | 41.49% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=6000, fold=2, folds=5, l=2, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 44.10% | 84.94% | 46.47% | 97.85% | 36.92% | 99.43% | 62.70% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=6500, fold=2, folds=5, l=2, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 38.29% | 84.63% | 43.61% | 97.16% | 43.55% | 98.54% | 43.64 |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=2, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 36.79% | 84.02% | 42.46% | 97.10% | 42.38% | 98.52% | 42.54% | 
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=5500, fold=2, folds=5, l=3, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 35.52% | 85.12% | 39.64% | 96.94% | 39.83% | 98.42% | 39.46% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=6000, fold=2, folds=5, l=3, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 40.32% | 84.81% | 42.98% | 97.66% | 34.89% | 99.29% | 55.96% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=6500, fold=2, folds=5, l=3, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 41.05% | 84.70% | 43.89% | 97.78% | 34.40% | 99.42% | 60.62% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=3, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 36.15% | 84.38% | 41.16% | 97.03% | 41.16% | 98.47% | 41.17% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=5500, fold=2, folds=5, l=4, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 35.39% | 85.20% | 39.29% | 96.94% | 39.20% | 98.44% | 39.38% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=6000, fold=2, folds=5, l=4, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 35.63% | 84.96% | 39.94% | 96.98% | 39.83% | 98.45% | 40.06% |
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=6500, fold=2, folds=5, l=4, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 35.71% | 84.83% | 40.05% | 96.98% | 39.99% | 98.45% | 40.12% |
| ####### PARAMETERS ####### Namespace(aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=2, folds=5, l=4, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 36.26% | 84.68% | 40.87% | 97.02% | 40.80% | 98.47% | 40.93% |


l: {1, 2, 3, 4} => 1
epochs: {4000, 5000, 6000, 7000, 8000} => 7000
n: {16, 32, 64, 96 } => 64
aggregator = {mul, sum, mean, concatenate} => mul
lr={0.01, 0.001, 0.005, 0.0001} => 0.001