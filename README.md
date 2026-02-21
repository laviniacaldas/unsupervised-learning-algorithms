# Unsupervised Learning Algorithms

## Overview

This repository contains from-scratch implementations of classical
unsupervised learning and probabilistic modeling algorithms using Python
and NumPy.

The objective is to explore the mathematical foundations, numerical
behavior, and algorithmic structure of clustering and statistical
sequence models without relying on high-level machine learning
libraries.

------------------------------------------------------------------------

## Implemented Algorithms

### 1. K-Means Clustering

K-Means partitions a dataset into $K$ clusters by minimizing the
within-cluster sum of squared distances:

$$
\underset{\{\mu_k\}}{\arg\min} \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
$$

Where:

-   $\mu_k$ is the centroid of cluster $k$
-   $C_k$ is the set of samples assigned to cluster $k$

Assignment step:

$$
C_k = \{ x_i : \|x_i - \mu_k\|^2 \leq \|x_i - \mu_j\|^2, \forall j \}
$$

Update step:

$$
\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
$$

------------------------------------------------------------------------

### 2. Gaussian Mixture Model (GMM)

A Gaussian Mixture Model represents the data distribution as a weighted
sum of Gaussian components:

$$
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$

Where:

-   $\pi_k$ are mixture weights, such that $\sum_{k=1}^{K} \pi_k = 1$
-   $\mu_k$ is the mean vector
-   $\Sigma_k$ is the covariance matrix

Parameter estimation is performed using the Expectation-Maximization
(EM) algorithm.

E-step (Responsibilities):

$$
\gamma_{ik} = 
\frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}
{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}
$$

M-step (Parameter Updates)

Mixture weights:

$$
\pi_k = \frac{1}{N} \sum_{i=1}^{N} \gamma_{ik}
$$

Means:

$$
\mu_k = 
\frac{\sum_{i=1}^{N} \gamma_{ik} x_i}
{\sum_{i=1}^{N} \gamma_{ik}}
$$

Covariances:

$$
\Sigma_k = 
\frac{\sum_{i=1}^{N} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}
{\sum_{i=1}^{N} \gamma_{ik}}
$$

Log-likelihood maximized at each iteration:

$$
\mathcal{L} = 
\sum_{i=1}^{N} 
\log \left( 
\sum_{k=1}^{K} 
\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) 
\right)
$$

------------------------------------------------------------------------

### 3. Hidden Markov Model (HMM)

A Hidden Markov Model is defined by:

-   Hidden states $S = \{s_1, \dots, s_K\}$
-   Transition probabilities $A_{ij} = P(s_{t+1}=j \mid s_t=i)$
-   Emission probabilities $B_j(x) = P(x_t \mid s_t=j)$
-   Initial state distribution $\pi$

Joint probability of a state sequence $S$ and observation sequence $X$:

$$
P(X, S) = 
\pi_{s_1} 
\prod_{t=2}^{T} A_{s_{t-1}, s_t}
\prod_{t=1}^{T} B_{s_t}(x_t)
$$

Learning is typically performed via the Baum-Welch algorithm, a special
case of EM.

------------------------------------------------------------------------

## Mathematical Focus

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
