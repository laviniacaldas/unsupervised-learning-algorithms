# Unsupervised Learning Algorithms

## Overview

This repository contains from-scratch implementations of classical
unsupervised learning and probabilistic modeling algorithms using Python
and NumPy.

The primary objective of this project is to explore the mathematical
foundations, numerical behavior, and algorithmic structure of clustering
and statistical sequence models without relying on high-level machine
learning libraries.

The implementations emphasize clarity, mathematical transparency, and
direct correspondence with theoretical formulations.

------------------------------------------------------------------------

## Implemented Algorithms

### 1. K-Means Clustering

K-Means partitions a dataset into ( K ) clusters by minimizing the
within-cluster sum of squared distances:

\[ `\underset{\{\mu_k\}}{\arg\min}`{=tex} `\sum`{=tex}*{k=1}\^{K}
`\sum`{=tex}*{x_i `\in `{=tex}C_k} \| x_i - `\mu`{=tex}\_k \|\^2 \]

Where:

-   ( `\mu`{=tex}\_k ) is the centroid of cluster ( k )
-   ( C_k ) is the set of samples assigned to cluster ( k )

The algorithm iteratively alternates between:

1.  Assignment step\
    \[ C_k = { x_i : \|x_i - `\mu`{=tex}\_k\|\^2 `\leq `{=tex}\|x_i -
    `\mu`{=tex}\_j\|\^2, `\forall `{=tex}j } \]

2.  Update step\
    \[ `\mu`{=tex}*k = `\frac{1}{|C_k|}`{=tex} `\sum`{=tex}*{x_i
    `\in `{=tex}C_k} x_i \]

------------------------------------------------------------------------

### 2. Gaussian Mixture Model (GMM)

A Gaussian Mixture Model represents the data distribution as a weighted
sum of Gaussian components:

\[ p(x) = `\sum`{=tex}\_{k=1}\^{K} `\pi`{=tex}\_k `\mathcal{N}`{=tex}(x
`\mid `{=tex}`\mu`{=tex}\_k, `\Sigma`{=tex}\_k) \]

Where:

-   ( `\pi`{=tex}*k ) are mixture weights, such that (
    `\sum`{=tex}*{k=1}\^{K} `\pi`{=tex}\_k = 1 )
-   ( `\mu`{=tex}\_k ) is the mean vector
-   ( `\Sigma`{=tex}\_k ) is the covariance matrix

Parameter estimation is performed using the Expectation-Maximization
(EM) algorithm.

### E-Step (Responsibilities)

\[ `\gamma`{=tex}*{ik} =
`\frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}`{=tex}
{`\sum`{=tex}*{j=1}\^{K} `\pi`{=tex}\_j `\mathcal{N}`{=tex}(x_i
`\mid `{=tex}`\mu`{=tex}\_j, `\Sigma`{=tex}\_j)} \]

### M-Step (Parameter Updates)

Mixture weights: \[ `\pi`{=tex}*k = `\frac{1}{N}`{=tex}
`\sum`{=tex}*{i=1}\^{N} `\gamma`{=tex}\_{ik} \]

Means: \[ `\mu`{=tex}*k = `\frac{\sum_{i=1}^{N} \gamma_{ik} x_i}`{=tex}
{`\sum`{=tex}*{i=1}\^{N} `\gamma`{=tex}\_{ik}} \]

Covariances: \[ `\Sigma`{=tex}*k =
`\frac{\sum_{i=1}^{N} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}`{=tex}
{`\sum`{=tex}*{i=1}\^{N} `\gamma`{=tex}\_{ik}} \]

The log-likelihood maximized at each iteration is:

\[ `\mathcal{L}`{=tex} = `\sum`{=tex}*{i=1}\^{N}
`\log `{=tex}`\left`{=tex}( `\sum`{=tex}*{k=1}\^{K} `\pi`{=tex}\_k
`\mathcal{N}`{=tex}(x_i `\mid `{=tex}`\mu`{=tex}\_k, `\Sigma`{=tex}\_k)
`\right`{=tex}) \]

------------------------------------------------------------------------

### 3. Hidden Markov Model (HMM)

A Hidden Markov Model is defined by:

-   A set of hidden states ( S = {s_1, `\dots`{=tex}, s_K} )
-   Transition probabilities ( A\_{ij} = P(s\_{t+1}=j
    `\mid `{=tex}s_t=i) )
-   Emission probabilities ( B_j(x) = P(x_t `\mid `{=tex}s_t=j) )
-   Initial state distribution ( `\pi `{=tex})

The joint probability of a state sequence ( S ) and observation sequence
( X ) is:

\[ P(X, S) = `\pi`{=tex}*{s_1} `\prod`{=tex}*{t=2}\^{T} A\_{s\_{t-1},
s_t} `\prod`{=tex}*{t=1}\^{T} B*{s_t}(x_t) \]

Learning is typically performed via the Baum-Welch algorithm, a special
case of EM.

------------------------------------------------------------------------

## Mathematical Focus

The project emphasizes:

-   Maximum Likelihood Estimation (MLE)
-   Expectation-Maximization framework
-   Covariance estimation and numerical stability
-   Log-likelihood tracking and convergence behavior
-   Probabilistic interpretation of clustering

------------------------------------------------------------------------

## Repository Structure

GMM.py -- Gaussian Mixture Model implementation\
GMM_HMM.py -- Hidden Markov Model implementation\
aplicacao_GMM.py -- GMM usage example\
aplicacao_kmeans.py -- K-Means usage example

------------------------------------------------------------------------

## Technical Stack

-   Python
-   NumPy
-   Matplotlib

------------------------------------------------------------------------

## Research and Extension Directions

Potential future developments include:

-   Numerical stabilization using log-sum-exp
-   Vectorized implementations for computational efficiency
-   Model selection using AIC/BIC
-   Comparison with scikit-learn implementations
-   Extension to supervised and semi-supervised variants
-   Integration with real-world datasets

------------------------------------------------------------------------

## Purpose

This repository serves as:

-   A study resource for probabilistic machine learning
-   A reference implementation aligned with theoretical formulations
-   A foundation for advanced research in statistical modeling and
    sequence analysis
