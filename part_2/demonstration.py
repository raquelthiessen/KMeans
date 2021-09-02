
import matplotlib.pyplot as plt
import numpy as np
from davies_bouldin_index import DaviesBouldinIndex

data = np.loadtxt("part_2/cluster_validation_data.txt", delimiter=",")
db_index = DaviesBouldinIndex(data)
db_index.model_selection()

# Plot the data
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(8, 6), sharex=False, sharey=False
)

# Set up Plot
fig.suptitle(f"Determining Clusters using Davies Bouldin", fontsize=16)
fte_colors = {
    0: "#E9BFFF",
    1: "#BFF3FF",
    2: "#FFD4BF",
    3: "#242AB3",
    4: "#019C2E",
    5: "#A02189",
}

# The final k-means-Euclidean plot
km_e_colors = [fte_colors[label] for label in db_index.labels]
ax1.scatter(data[:, 0], data[:, 1], c=km_e_colors)
ax1.set_title(
    f"k-means-Euclidean", fontdict={"fontsize": 12}
)

ax2.plot(list(range(2,db_index.max_iterations+2)), db_index.indices)
ax2.set_title(
    f"Davies Bouldin Index", fontdict={"fontsize": 12}
)
ax2.set_xlabel("Number of Clusters")
ax2.set_ylabel("Davies Bouldin Index")

plt.show()