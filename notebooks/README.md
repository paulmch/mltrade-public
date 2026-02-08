# Notebooks

Experimental Jupyter notebooks from the V2 development phase. Cell outputs have been stripped for cleanliness -- only the code and markdown cells remain.

---

## NDXkan.ipynb -- Kolmogorov-Arnold Network Experiments

Experiments with Kolmogorov-Arnold Networks (KAN) as an alternative architecture for NASDAQ-100 direction prediction. KAN replaces the fixed activation functions of traditional neural networks (ReLU, sigmoid) with learnable activation functions on the network edges, based on the Kolmogorov-Arnold representation theorem.

This notebook explored whether KAN's ability to learn arbitrary univariate functions on each connection could capture nonlinear relationships in financial features more effectively than dense layers. The experiments informed the eventual architecture choice (NCP/LTC was selected for the production system).

---

## UpDownML.ipynb -- Early ML Experiments

Early-stage experiments with up/down prediction models for index movement. This notebook represents the initial exploration phase before the system architecture was established -- testing basic classification approaches for predicting whether the index would move up or down on the next trading day.

These experiments helped establish the feature engineering patterns and evaluation methodology that carried forward into the production system.
