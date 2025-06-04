from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)

inertia = [KMeans(n_clusters=i).fit(X).inertia_ for i in range(1, 11)]

plt.plot(range(1, 11), inertia, "bo-")
plt.xlabel("Clusters")
plt.ylabel("Inertia")
plt.show()

k_means = KMeans(n_clusters=3)
k_means.fit(X)

centroid = k_means.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=k_means.labels_)
plt.scatter(centroid[:, 0], centroid[:, 1], c="r", s=200, marker="X")

plt.title("K_means Clustering")
plt.show()
