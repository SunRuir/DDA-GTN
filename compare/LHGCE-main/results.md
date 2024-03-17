| parameters | aupr | auc | f1 | accuracy | recall | specificity | precision | 
| - | - | - | - | - | - | - | - | 
| aggregator='mul', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=-1, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 56% | 84.65% | 65.50% | 98.45% | 56.06% | 99.55% | 82.39% |
| aggregator='sum', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=0, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 37.90% | 84.48% | 41.94% | 97.32% | 38.01% | 98.86% | 48.96% |
| aggregator='mean', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=-1, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 37.03% | 84.97% | 40.46% | 97.33% | 35.86% | 98.91% | 47.81% | 
| aggregator='concatenate', dataset='LAGCN', encoder='SAGE', epochs=7000, fold=3, folds=5, l=1, lr_encoder=0.001, lr_linear=0.001, n=64, negative_split='all', same=False, thr_percent=2.5 | 44.55% | 83.49% | 50.46% | 97.87% | 41.92% | 99.31% | 68.58% |

[0.44557967 0.83493664 0.50460611 0.97872139 0.41927776 0.99318614
 0.68586446]
