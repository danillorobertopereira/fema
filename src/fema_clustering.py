import sys
import os
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from numba import jit
from scipy.spatial.distance import cdist

sys.path.append('C:\\Users\\coton\\Desktop\\github\\fema\\src\\')
from fem_basis import Basis  
import fema_classifier

class FEMaClustering:
    def __init__(self, z: float = 2, basis=Basis.shepardBasis):
        self.z = z
        self.dim = None
        self.qtd_samples = None
        self.dist_matrix = None
        self.conquest = None
        self.conquested = None
        self.splited = None
        self.samples = None
        self.random_samples = None
        self.all_samples = None
        self.labels = None
        self.basis = basis
        self.qtd_diff_samples = None     
        self.th_same_cluster = None
        self.min_distance = None
        self.model = None
        self.expand_matrix = None

    def plot_points(self, points: np.ndarray, labels: np.ndarray = None):
        if points.shape[1] == 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, marker='o', edgecolors='black')
            if labels is not None:
                legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
                ax.add_artist(legend1)
            ax.grid(True)
            plt.show()
        elif points.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, marker='o', edgecolors='black')
            if labels is not None:
                legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
                ax.add_artist(legend1)
            ax.grid(True)
            plt.show()
        else:
            print("Plotting is only supported for 2D and 3D points.")

    def calculate_matrices(self, points: np.ndarray):
        N = len(points)
        self.dist_matrix = np.zeros((N, N))
        
        for ind, p in enumerate(points):
            dist = np.linalg.norm(points - p, axis=1)
            self.dist_matrix[ind, :] = dist
            self.dist_matrix[ind, ind] = np.inf  # Ignore self-distance
        
    def generate_random_points(self, bounds: List[Tuple[float, float]], num_points: int) -> np.ndarray:
        points = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (num_points, len(bounds)))
        return points

    def filter_nearby_points(self, random_points: np.ndarray, reference_points: np.ndarray, min_distance: float) -> np.ndarray:
        mask = np.all(np.linalg.norm(reference_points[:, np.newaxis, :] - random_points, axis=2) >= min_distance, axis=0)
        return random_points[mask]

    def lists_to_np_array(self, lists: List[List[float]]) -> np.ndarray:
        return np.array(lists)

    """def atribute_all_samples_and_labels(self, samples: np.ndarray, qtd_samples: int):
        self.samples = samples
        self.dim = samples.shape[1]
        self.qtd_samples = samples.shape[0]

        bounds = [(min(samples[:, i]), max(samples[:, i])) for i in range(self.dim)]
        self.bounds = bounds.copy()

        random_points = self.generate_random_points(bounds=bounds, num_points=qtd_samples)
        self.random_samples = self.filter_nearby_points(random_points=random_points, reference_points=self.samples, min_distance=self.min_distance)

        self.all_samples = np.concatenate((self.samples, self.random_samples))
        self.labels = np.zeros(self.all_samples.shape[0], dtype=int)
        self.labels[:samples.shape[0]] = 1
    """

    def atribute_all_samples_and_labels(self, samples: np.ndarray, qtd_samples: int):
        self.samples = samples
        self.dim = samples.shape[1]
        self.qtd_samples = samples.shape[0]

        bounds = [(min(samples[:, i]), max(samples[:, i])) for i in range(self.dim)]
        self.bounds = bounds.copy()

        def generate_random_point():
            return np.array([np.random.uniform(low, high) for low, high in bounds])

        random_samples = []
        
        for _ in range(qtd_samples):
            max_min_distance = -np.inf
            best_point = None
            
            # Sample a large number of candidate points
            candidate_points = np.array([generate_random_point() for _ in range(10*qtd_samples)])
            
            # Calculate distances from candidate points to all existing and new points
            all_points = np.vstack([samples] + random_samples)
            distances = cdist(candidate_points, all_points, metric='euclidean')
            
            # Find the candidate point that maximizes the minimum distance to all existing points
            min_distances = distances.min(axis=1)
            best_idx = min_distances.argmax()
            best_point = candidate_points[best_idx]
            
            random_samples.append(best_point)
    
            self.random_samples = random_samples.copy()

            self.all_samples = np.concatenate((self.samples, self.random_samples))
            self.labels = np.zeros(self.all_samples.shape[0], dtype=int)
            self.labels[:samples.shape[0]] = 1

    
    def expand_adjacency_matrix(self, ref_matrix: np.ndarray):
        N = ref_matrix.shape[0]
        self.expanded_matrix = np.copy(ref_matrix)

        for i in range(N):
            for j in range(N):
                if ref_matrix[i, j] != 0:
                    self.expanded_matrix[i, j] = 1
                    self.expanded_matrix[i, :] = np.logical_or(self.expanded_matrix[i, :], ref_matrix[j, :]).astype(int)

    def label_connected_components(self):
        visited = np.zeros(self.qtd_samples, dtype=bool)
        labels = np.full(self.qtd_samples, -1, dtype=int)
        current_label = 0

        for vertex in range(self.qtd_samples):
            if not visited[vertex]:
                bfs_queue = deque([vertex])
                visited[vertex] = True

                while bfs_queue:
                    current_vertex = bfs_queue.popleft()
                    labels[current_vertex] = current_label

                    neighbors = np.where(self.expanded_matrix[current_vertex] == 1)[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            bfs_queue.append(neighbor)
                            visited[neighbor] = True

                current_label += 1

        self.labels = labels
        return self.labels

    def fit(self, samples: np.ndarray, min_distance: float = 10, qtd_samples_perc: float  = 1.0):
        self.min_distance = min_distance
        self.qtd_samples = samples.shape[0]
        self.dim = samples.shape[1]

        self.atribute_all_samples_and_labels(samples=samples,qtd_samples=int(self.qtd_samples*qtd_samples_perc))
        self.calculate_matrices(self.samples)

        self.model = fema_classifier.FEMaClassifier(basis=self.basis)
        self.model.fit(self.all_samples, self.labels.reshape((len(self.labels), 1)))

    def predict(self, th_same_cluster: float = 0.75, qtd_diff_samples: float = 50):
        self.qtd_diff_samples = qtd_diff_samples
        self.th_same_cluster = th_same_cluster

        self.dist_matrix = np.zeros((self.qtd_samples, self.qtd_samples))
        diff_factor = 1 / (self.qtd_diff_samples - 1)

        for i in range(self.qtd_samples):
            for j in range(i + 1, self.qtd_samples):
                diff = (self.samples[i] - self.samples[j]) * diff_factor
                test_samples = np.array([self.samples[i] - diff * k for k in range(self.qtd_diff_samples)])
                _, prob = self.model.predict(test_samples, 3)
                if np.any(prob[:, 1] < th_same_cluster):
                    self.dist_matrix[i, j] = 0
                    self.dist_matrix[j, i] = 0
                else:
                    self.dist_matrix[i, j] = 1
                    self.dist_matrix[j, i] = 1

        self.expand_adjacency_matrix(self.dist_matrix)
        for _ in range(10):
            self.expand_adjacency_matrix(self.expanded_matrix)

        self.labels = self.label_connected_components()
        return self.labels
