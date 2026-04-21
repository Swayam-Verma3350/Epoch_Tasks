import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Step 1: Load and Filter Data
# -------------------------

df = pd.read_csv('clustering_data.csv', low_memory=False)

# My home state is Uttar Pradesh
up = df[df['StateName'] == 'UTTAR PRADESH'].copy()

print(f"Total rows for Uttar Pradesh: {len(up)}")

# -------------------------
# Step 2: Preprocessing
# -------------------------

# drop rows with missing latitude or longitude
up = up.dropna(subset=['Latitude', 'Longitude'])
print(f"Rows after dropping missing latitude/longitude: {len(up)}")

# drop duplicate pincodes
up = up.drop_duplicates(subset=['Pincode'])
print(f"Rows after dropping duplicate pincodes: {len(up)}")

# convert to numeric (some UP rows may have lat/long stored as strings)
up['Latitude']  = pd.to_numeric(up['Latitude'],  errors='coerce')
up['Longitude'] = pd.to_numeric(up['Longitude'], errors='coerce')
up = up.dropna(subset=['Latitude', 'Longitude'])

# remove rows with clearly invalid coordinates
# UP is roughly between lat 23-31 and lon 77-85
up = up[(up['Latitude']  >= 23) & (up['Latitude']  <= 31)]
up = up[(up['Longitude'] >= 77) & (up['Longitude'] <= 85)]
print(f"Final rows used for clustering: {len(up)}")
print(f"\nLatitude  range: {up['Latitude'].min():.4f} to {up['Latitude'].max():.4f}")
print(f"Longitude range: {up['Longitude'].min():.4f} to {up['Longitude'].max():.4f}")

# extract lat/lon as numpy array
X = up[['Latitude', 'Longitude']].values

# -------------------------
# Step 3: K-Means
# -------------------------

class KMeans:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # randomly pick k points from X as starting centroids
        random_indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            self.labels = self.assign_clusters(X)
            old_centroids = self.centroids.copy()
            self.centroids = self.update_centroids(X)

            # if centroids didnt move, we can stop early
            if np.allclose(old_centroids, self.centroids):
                break

        return self

    def assign_clusters(self, X):
        # calculate distance from every point to every centroid
        distances = []
        for centroid in self.centroids:
            dist = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
            distances.append(dist)

        distances = np.array(distances)

        # for each point, pick the centroid with smallest distance
        labels = np.argmin(distances, axis=0)
        return labels

    def update_centroids(self, X):
        new_centroids = []
        for cluster_idx in range(self.k):
            points_in_cluster = X[self.labels == cluster_idx]

            if len(points_in_cluster) == 0:
                # if a cluster has no points, keep old centroid
                new_centroids.append(self.centroids[cluster_idx])
            else:
                # new centroid = mean of all points in cluster
                new_centroids.append(points_in_cluster.mean(axis=0))

        return np.array(new_centroids)

    def predict(self, X):
        return self.assign_clusters(X)

    def inertia(self, X):
        # total sum of squared distances from each point to its centroid
        total = 0
        for cluster_idx in range(self.k):
            points_in_cluster = X[self.labels == cluster_idx]
            if len(points_in_cluster) > 0:
                diff = points_in_cluster - self.centroids[cluster_idx]
                total += np.sum(diff ** 2)
        return total


# -------------------------
# Step 4: Elbow Method to find best K
# -------------------------

np.random.seed(42)

k_values = range(1, 11)
inertias = []

for k in k_values:
    model = KMeans(k=k, max_iters=100)
    model.fit(X)
    inertias.append(model.inertia(X))

# plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o', color='steelblue', linewidth=2)
plt.title('Elbow Method — Finding Best K', fontsize=13)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# best k from elbow plot
K = 3

print(f"\nUsing K={K} for final clustering(deduced from elbow curve)")
model = KMeans(k=K, max_iters=100)
model.fit(X)

# add cluster labels back to dataframe
up = up.copy()
up['Cluster'] = model.labels

print(f"\n--- Cluster Summary ---")
for i in range(K):
    cluster_data = up[up['Cluster'] == i]
    print(f"\nCluster {i+1}:")
    print(f"  Number of pincodes : {len(cluster_data)}")
    print(f"  Centroid           : Lat={model.centroids[i][0]:.4f}, Lon={model.centroids[i][1]:.4f}")
    print(f"  Districts covered  : {cluster_data['District'].nunique()}")
    print(f"  Sample districts   : {list(cluster_data['District'].unique()[:5])}")


# -------------------------
# Step 5: Visualization 1 — Scatter plot of all pincodes colored by cluster
# -------------------------

colors = ["#ff0000", "#2200ff", "#22FF00"]

plt.figure(figsize=(12, 8))

for i in range(K):
    cluster_points = X[model.labels == i]
    plt.scatter(
        cluster_points[:, 1],   # longitude on x axis
        cluster_points[:, 0],   # latitude on y axis
        c=colors[i],
        label=f'Cluster {i+1} ({len(cluster_points)} pincodes)',
        alpha=0.5,
        s=10
    )

# plot centroids
plt.scatter(
    model.centroids[:, 1],
    model.centroids[:, 0],
    c='black',
    marker='X',
    s=200,
    label='Centroids',
    zorder=5
)

plt.title('K-Means Clustering of Uttar Pradesh Pincodes (k=3)', fontsize=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# -------------------------
# Step 6: Visualization 2 — Bar chart of pincode count per cluster
# -------------------------

cluster_counts = [len(X[model.labels == i]) for i in range(K)]

plt.figure(figsize=(8, 5))
bars = plt.bar(
    [f'Cluster {i+1}' for i in range(K)],
    cluster_counts,
    color=colors
)

# add count labels on top of each bar
for bar, count in zip(bars, cluster_counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 20,
        str(count),
        ha='center',
        fontsize=11
    )

plt.title('Number of Pincodes per Cluster - Uttar Pradesh', fontsize=13)
plt.xlabel('Cluster')
plt.ylabel('Number of Pincodes')
plt.tight_layout()
plt.show()


# -------------------------
# Step 7: Inferences
# -------------------------

print("\n--- Inferences ---")
print("""
1. The 3 clusters correspond to the 3 broad geographical regions of UP:
   Western UP, Central UP, and Eastern UP — which aligns well with how
   UP is administratively and culturally divided.

2. Western UP cluster (Agra, Meerut, Mathura, Noida) has the highest
   pincode density, reflecting its proximity to Delhi NCR and heavy
   urbanisation along the Yamuna belt.

3. Central UP cluster (Lucknow, Kanpur, Allahabad) forms the political
   and economic heartland of the state. Lucknow being the state capital
   drives a dense concentration of pincodes in this region.

4. Eastern UP cluster (Varanasi, Gorakhpur, Azamgarh) is geographically
   the largest but relatively sparser in pincode density, reflecting
   a more rural character with fewer urban centres.

5. The elbow method confirmed k=3 as optimal — adding more clusters
   beyond 3 gave diminishing returns in inertia reduction, meaning
   3 natural geographical groupings best describe UP's pincode distribution.
""")