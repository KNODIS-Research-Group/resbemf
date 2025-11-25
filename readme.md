# Restricted Bernoulli Matrix Factorization: Balancing the trade-off between prediction accuracy and coverage in Classification based collaborative filtering

This repository includes the source code for the experiments discussed in the manuscript titled *Restricted Bernoulli Matrix Factorization: Balancing the trade-off between prediction accuracy and coverage in Classification based collaborative filtering*. The manuscript has been submitted to the *International Journal of Interactive Multimedia and Artificial Intelligence* journal. You can access the preprint on [arXiv](https://arxiv.org/abs/2210.10619).

## Random search results

MovieLens 100K:

![Hyper-parameters](figs/ml100k-hyperparameters.png)

MovieLens 1M:

![Hyper-parameters](figs/ml1m-hyperparameters.png)

FilmTrust:

![Hyper-parameters](figs/ft-hyperparameters.png)

MyAnimeList:

![Hyper-parameters](figs/anime-hyperparameters.png)

MyAnimeList:

![Hyper-parameters](figs/ml10m-hyperparameters.png)

## Pareto front:

![Pareto front](figs/pareto-front.png)

## Test error:

Matrix factorization based collaborative filtering:

![Test error](figs/mf-test-error.png)

Artificial neural network based collaborative filtering:

![Test error](figs/nn-test-error.png)

> To reproduce the experiments from this research, unzip the prediction files into the preds/gmcm and preds/mwgp directories.

