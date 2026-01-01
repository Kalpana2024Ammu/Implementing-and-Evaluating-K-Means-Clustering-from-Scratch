# Implementing-and-Evaluating-K-Means-Clustering-from-Scratch
.
# 🔍 Custom K-Means Clustering (From Scratch)

## This project implements a Custom K-Means clustering algorithm from scratch using NumPy and compares its performance with Scikit-learn’s KMeans. The goal is to understand the internal working of K-Means and evaluate clustering quality and execution time.

# 🎯 Project Objectives

## Implement the K-Means clustering algorithm without using built-in clustering methods

## Support different centroid initialization techniques

## Compare clustering results with Scikit-learn’s implementation

## Visualize and analyze clustering performance

# ⚙️ Algorithm Implementation

## The custom K-Means implementation includes:

## Configurable number of clusters (n_clusters)

## Maximum iterations and convergence tolerance

## Centroid initialization methods:

## Random initialization

## K-Means++ initialization

## Inertia calculation to measure cluster compactness

## Iterative centroid updates until convergence

# 📊 Dataset

## Synthetic dataset generated using make_blobs

## Number of samples: 500

## Number of clusters: 4

## Used for controlled evaluation of clustering performance

# 🔁 Multiple Runs & Best Model Selection

## The model is trained multiple times with different random seeds

## For each run, the following are recorded:

## Inertia

## Execution time

## Cluster labels

## Centroids

## The best run is selected based on minimum inertia

# 📈 Performance Comparison

## Compared Custom K-Means with Scikit-learn KMeans

## Evaluation metrics:

## Inertia

## Execution Time

## Results demonstrate how close a custom implementation can perform relative to a production-ready library

# 📊 Visualization

## Scatter plots showing:

## Cluster assignments

## Centroid positions

## Side-by-side comparison:

## Custom K-Means (Best Run)

## Scikit-learn K-Means

# 🛠️ Technologies & Libraries

## Python

## NumPy

## Matplotlib

## Scikit-learn

# 🎯 Conclusion

## This project provides a hands-on understanding of the K-Means clustering algorithm, highlighting the impact of centroid initialization, convergence criteria, and multiple runs on clustering quality, while validating results against Scikit-learn’s implementation.
