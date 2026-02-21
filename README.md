# Unsupervised Learning Algorithms

## Overview

This repository contains from-scratch implementations of classical
unsupervised learning and probabilistic modeling algorithms using Python
and NumPy.

The primary objective of this project is to explore the mathematical
foundations, numerical behavior, and algorithmic structure of clustering
and statistical sequence models without relying on high-level machine
learning libraries such as scikit-learn.

The implementations emphasize clarity, mathematical transparency, and
direct correspondence with theoretical formulations.

------------------------------------------------------------------------

## Implemented Algorithms

### 1. K-Means Clustering

A centroid-based clustering algorithm that partitions data into *K*
clusters by minimizing the within-cluster sum of squared distances:

argmin Σ \|\|x - μ_k\|\|²

Characteristics: - Distance-based clustering - Hard assignments -
Iterative refinement via Lloyd's algorithm - Sensitive to initialization

------------------------------------------------------------------------

### 2. Gaussian Mixture Model (GMM)

A probabilistic clustering approach that models the data distribution as
a weighted sum of Gaussian components:

p(x) = Σ_k π_k N(x \| μ_k, Σ_k)

Parameters are estimated using the Expectation-Maximization (EM)
algorithm:

-   E-step: Compute responsibilities (posterior probabilities)
-   M-step: Update mixture weights, means, and covariance matrices
-   Log-likelihood maximization at each iteration

Characteristics: - Soft clustering - Full covariance modeling -
Probabilistic interpretation - More expressive than K-Means

------------------------------------------------------------------------

### 3. Hidden Markov Model (HMM)

A generative statistical model for sequential data defined by:

-   Hidden states
-   Transition probabilities
-   Emission probabilities

Key algorithms typically involved: - Forward algorithm - Backward
algorithm - Baum-Welch (EM for HMM) - Viterbi decoding

This implementation provides a foundation for statistical sequence
modeling and probabilistic temporal reasoning.

------------------------------------------------------------------------

## Mathematical Focus

The project emphasizes:

-   Maximum likelihood estimation
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
