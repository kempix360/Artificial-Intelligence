import numpy as np


def initialize_centroids_forgy(data, k):
    # TODO implement random initialization

    centroid_indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[centroid_indices]

    return centroids


def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initialization

    centroids = np.zeros((k, data.shape[1]))
    centroids[0] = data[np.random.choice(data.shape[0], 1)]

    for i in range(1, k):
        distances = np.zeros((data.shape[0], i))
        for j in range(i):
            distances[:, j] = np.linalg.norm(data - centroids[j], axis=1)  # calculate euclidean norm for each vector

        max_distances = 0
        index = 0
        for j in range(distances.shape[0]):
            if np.sum(distances[j]) > max_distances:
                max_distances = np.sum(distances[j])
                index = j

        centroids[i] = data[index]

    return centroids


def assign_to_cluster(data, centroids):
    # TODO find the closest cluster for each data point
    assignments = np.zeros(data.shape[0], dtype=int)
    for i in range(data.shape[0]):
        differences = np.linalg.norm(data[i] - centroids, axis=1)
        assignments[i] = np.argmin(differences)  # return index of the smallest value in array

    return assignments


def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments

    centroids = np.zeros((len(np.unique(assignments)), data.shape[1]))
    for idx in range(len(np.unique(assignments))):
        centroids[idx] = np.mean(data[assignments == idx], axis=0)
    return centroids


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :]) ** 2))


def k_means(data, num_centroids, kmeansplusplus):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100):  # max number of iteration = 100
        # print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)
