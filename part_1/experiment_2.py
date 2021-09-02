import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from k_means import KMeans

# Set up data
features, true_labels = make_blobs(
    n_samples=500,
    n_features=2,
    cluster_std=1,
    random_state=150
)

# Initialize KMeans
kmeans_euclidean = KMeans(n_clusters=3, distance="euclidean")
kmeans_mahalanobis = KMeans(n_clusters=3, distance="mahalanobis")

# Fit to the features
kmeans_euclidean.fit(features)
kmeans_mahalanobis.fit(features)

# Plot the data
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(8, 6), sharex=True, sharey=True
)
fte_colors = {
    0: "#E9BFFF",
    1: "#BFF3FF",
    2: "#FFD4BF",
}

#The k-means-Euclidean plot
km_e_colors = [fte_colors[label] for label in kmeans_euclidean.labels]
ax1.scatter(features[:, 0], features[:, 1], c=km_e_colors)
ax1.set_title(
    f"k-means-Euclidean", fontdict={"fontsize": 12}
)

#The k-means-Mahalanobis plot
km_h_colors = [fte_colors[label] for label in kmeans_mahalanobis.labels]
ax2.scatter(features[:, 0], features[:, 1], c=km_h_colors)
ax2.set_title(
    f"k-means-Mahalanobis", fontdict={"fontsize": 12}
)

#The k-means-Ground Truth plot
km_g_colors = [fte_colors[label] for label in true_labels]
ax3.scatter(features[:, 0], features[:, 1], c=km_g_colors)
ax3.set_title(
    f"Ground Truth", fontdict={"fontsize": 12}
)

plt.show()