import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
#3 Define Lentgh of distance
def define_distance(a, b):
    return np.linalg.norm(a - b)

#4 K-means clustering algorithm
def k_means_clustering(data, k, max_iterations=100):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [define_distance(point, centroid) for centroid in centroids]
            closest_centroid_idx = np.argmin(distances)
            clusters[closest_centroid_idx].append(point)
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return clusters


#5 Hierarchical clustering algorithm
def hierarchical_clustering(data, threshold):
    distances = euclidean_distances(data)
    clusters = [[point] for point in data]
    while len(clusters) > 1:
        np.fill_diagonal(distances, np.inf)
        min_distance_idx = np.unravel_index(np.argmin(distances), distances.shape)
        min_distance = distances[min_distance_idx]
        if min_distance > threshold:
            break
        merged_cluster = clusters[min_distance_idx[0]] + clusters[min_distance_idx[1]]
        del clusters[max(min_distance_idx)]
        del clusters[min(min_distance_idx)]
        clusters.append(merged_cluster)
        # Обновляем расстояния между кластерами
        new_distances = np.zeros((len(clusters), len(clusters)))
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distances_ij = [define_distance(a, b) for a in clusters[i] for b in clusters[j]]
                new_distances[i, j] = np.mean(distances_ij)
                new_distances[j, i] = new_distances[i, j]
        distances = new_distances
    return clusters

#7 Function to evaluate the average weighted size of clusters
def evaluate_cluster_sizes(clusters):
    cluster_sizes = [len(cluster) for cluster in clusters]
    weights = np.array(cluster_sizes) / sum(cluster_sizes)
    average_cluster_size = np.average(cluster_sizes, weights=weights)
    return average_cluster_size

# Display clusters in a separate window
def plot_clusters(clusters, title):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    plt.figure(figsize=(10, 6))
    plt.title(title)

    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i % len(colors)], label=f'Cluster {i+1}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

# Example dataset
data = [(0.1, 0.9), (0.3, 0.5), (0.7, 0.2), (0.8, 0.8), (0.4, 0.6),
        (0.2, 0.3), (0.6, 0.9), (0.9, 0.1), (0.5, 0.7), (0.3, 0.2),
        (0.1, 0.5), (0.7, 0.6), (0.4, 0.2), (0.8, 0.4), (0.6, 0.1),
        (0.2, 0.7), (0.9, 0.8), (0.5, 0.3), (0.3, 0.8), (0.1, 0.1)]

if __name__ == "__main__":
    # Number of clusters
    k = 3
    # Threshold for hierarchical clustering
    threshold = 0.5
    # Use randomly generated dataset or predefined dataset?
    RandData = True
    # Number of values in the random dataset
    RandNum = 1500
    #6 Enable hierarchical clustering method (extremely slow, recommended to use with datasets of 100-150 points maximum, otherwise wait for more than a minute)
    PlotHierarchical = True

    if RandData == True:
        data = [(round(random.random(), 3), round(random.random(), 3)) for _ in range(RandNum)]

    # K-means clustering
    k_means_clusters = k_means_clustering(np.array(data), k)
    print(f"[MAIN] Parameters: Random dataset: {RandData}, Threshold for hierarchical clustering: {threshold}, Number of clusters for K-means clustering: {k}")
    '''print("\nK-means clusters:")
    for i, cluster in enumerate(k_means_clusters):
        print(f"Cluster {i+1}: {cluster}")'''

    # Display K-means clusters
    plot_clusters(k_means_clusters, f'K-means Clustering with {RandNum} points')
   
    k_means_quality = evaluate_cluster_sizes(k_means_clusters)
    print(f"Average weighted size of clusters for K-means clustering: {k_means_quality}")

    if PlotHierarchical == True:
        # Hierarchical clustering
        hierarchical_clusters = hierarchical_clustering(np.array(data), threshold)
        '''print("\nHierarchical clusters:")
        for i, cluster in enumerate(hierarchical_clusters):
            print(f"Cluster {i+1}: {cluster}")'''

        hierarchical_quality = evaluate_cluster_sizes(hierarchical_clusters)
        print(f"Average weighted size of clusters for hierarchical clustering: {hierarchical_quality}")
        
        # Display hierarchical clusters
        plot_clusters(hierarchical_clusters, f'Hierarchical Clustering with {RandNum} points')
