import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time

class CustomKMeans:
    def __init__(self, n_clusters=4, max_iter=300, tol=1e-4, init="random", random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        if self.init == "random":
            return X[rng.choice(len(X), self.n_clusters, replace=False)]
        if self.init == "kmeans++":
            centroids = [X[rng.integers(len(X))]]
            for _ in range(1, self.n_clusters):
                distances = np.min(((X[:, None] - np.array(centroids))**2).sum(axis=2), axis=1)
                probs = distances / distances.sum()
                centroids.append(X[rng.choice(len(X), p=probs)])
            return np.array(centroids)
        raise ValueError("init must be 'random' or 'kmeans++'")

    def _inertia(self, X, centroids, labels):
        return np.sum((X - centroids[labels])**2)

    def fit(self, X):
        self.centroids_ = self._init_centroids(X)
        for _ in range(self.max_iter):
            distances = ((X[:, None] - self.centroids_)**2).sum(axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.sum((new_centroids - self.centroids_)**2) <= self.tol:
                break
            self.centroids_ = new_centroids
        self.labels_ = labels
        self.inertia_ = self._inertia(X, self.centroids_, labels)
        return self

X, true_labels = make_blobs(n_samples=500, centers=4, cluster_std=0.80, random_state=42)

N_RUNS = 10
custom_runs = []
for seed in range(N_RUNS):
    model = CustomKMeans(n_clusters=4, init="random", random_state=seed)
    start = time.time()
    model.fit(X)
    end = time.time()
    custom_runs.append({
        "seed": seed,
        "inertia": model.inertia_,
        "time": end - start,
        "labels": model.labels_,
        "centroids": model.centroids_
    })

best_run = min(custom_runs, key=lambda r: r["inertia"])

print("Custom K-Means runs:")
for r in custom_runs:
    print(f"Seed={r['seed']}  Inertia={r['inertia']:.2f}  Time={r['time']:.4f}s")

print("\nBest run:", best_run["seed"], "Inertia=", best_run["inertia"])

start = time.time()
sk = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
end = time.time()

sk_results = {
    "inertia": sk.inertia_,
    "time": end - start,
    "labels": sk.labels_,
    "centroids": sk.cluster_centers_
}

print("\nScikit-Learn:")
print(sk_results)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].scatter(X[:, 0], X[:, 1], c=best_run["labels"], s=20)
ax[0].scatter(best_run["centroids"][:, 0], best_run["centroids"][:, 1], c='red', s=200, marker='X')
ax[0].set_title("Custom K-Means (Best Run)")

ax[1].scatter(X[:, 0], X[:, 1], c=sk_results["labels"], s=20)
ax[1].scatter(sk_results["centroids"][:, 0], sk_results["centroids"][:, 1], c='red', s=200, marker='X')
ax[1].set_title("Scikit-Learn K-Means")

plt.tight_layout()
plt.show()
