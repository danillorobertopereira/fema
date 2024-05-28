import sys
import os
from typing import Tuple

from numpy.random.mtrand import random


sys.path.append('C:\\Users\\coton\\Desktop\\github\\fema\\src\\')


import math 
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from collections import deque

from fem_basis import Basis  
import fema_classifier


class FEMaClustering:
    def __init__(self, z: float = 2, sBasis=Basis.shepardBasis):
        self.z = z
        self.dim = None
        self.qtd_samples = None
        self.dist_matrix = None
        self.weight_matrix = None
        self.conquest = None
        self.conquested = None
        self.splited = None
        self.samples = None
        self.random_samples = None
        self.all_samples = None
        self.labels = None
        self.basis = Basis(z)
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
        self.weight_matrix = np.zeros((N, N))
        
        for (p, ind) in zip(points, range(N)):
            dist = np.array([np.linalg.norm(points[i] - p) for i in range(N)])
            self.dist_matrix[ind, :] = dist
            self.dist_matrix[ind, ind] = 10**20  # Large value to ignore self-distance
            self.weight_matrix[ind, :] = 1.0 / (self.dist_matrix[ind, :] ** self.z)
            self.weight_matrix[ind, :] /= sum(self.weight_matrix[ind, :] + 1**-20)

    def determine_conquest(self, points: np.ndarray):
        N = len(points)
        self.conquest = np.zeros(N)
        self.conquested = np.zeros(N)
        self.splited = np.zeros(N)

        for i in range(N):
            self.conquest[i] = np.argsort(self.weight_matrix[i, :])[N - 1]
            self.splited[i] = np.argsort(self.weight_matrix[i, :])[1]

        for i in range(N):
            if (i == int(self.conquest[int(self.conquest[i])])) and (i != int(self.splited[int(self.conquest[i])])):
                if i <= int(self.conquest[i]):
                    self.conquested[i] = i
                    self.conquested[int(self.conquest[i])] = i
                else:
                    self.conquested[i] = i
            else:
                self.conquested[i] = i

    def label_clusters(self, points: np.ndarray):
        self.plot_points(points, self.conquested)

    def generate_random_points(self, bounds: List[Tuple[float, float]], num_points: int) -> np.ndarray:
        """
        Generate random points within specified bounds for each dimension.

        Parameters:
        bounds (List[Tuple[float, float]]): List of tuples specifying min and max bounds for each dimension.
        num_points (int): Number of random points to generate.

        Returns:
        np.ndarray: Array of shape (num_points, len(bounds)) containing generated points.
        """
        points = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (num_points, len(bounds)))
        return points

    def filter_nearby_points(self, random_points: np.ndarray, reference_points: np.ndarray, min_distance: float) -> np.ndarray:
        """
        Filter points that are at least a minimum distance away from a set of reference points.

        Parameters:
        random_points (np.ndarray): Array of shape (num_points, dimensions) containing random points.
        reference_points (np.ndarray): Array of shape (num_reference_points, dimensions) containing reference points.
        min_distance (float): Minimum distance required.

        Returns:
        np.ndarray: Array of filtered points.
        """
        filtered_points = []
        for point in random_points:
            distances = np.linalg.norm(reference_points - point, axis=1)
            if np.all(distances >= min_distance):
                filtered_points.append(point)
        return np.array(filtered_points)

    def lists_to_np_array(self,lists: List[List[float]]) -> np.ndarray:
        """
        Convert a list of lists into a 2D NumPy array.

        Parameters:
        lists (List[List[float]]): List of lists of coordinates.

        Returns:
        np.ndarray: 2D array of coordinates.
        """
        return np.array(lists)

    def atribute_all_samples_and_labels(self, samples: np.ndarray):
        self.samples = samples
        self.dim = samples.shape[1]
        self.qtd_samples = samples.shape[0]

        bounds = []
        for i in range(self.dim):
            bounds.append((min(samples[:, i]), max(samples[:, i])))

        self.set_bounds(bounds=bounds) 

        self.weight_matrix = np.zeros((self.qtd_samples, self.qtd_samples))
        
        random_points = self.generate_random_points(bounds=bounds, num_points=2*self.qtd_samples)
        self.random_samples = self.filter_nearby_points(random_points=random_points, reference_points=self.samples, min_distance=self.min_distance)

        self.all_samples = np.concatenate((self.samples, self.random_samples))

        self.labels = np.zeros(self.all_samples.shape[0],dtype=int)
        self.labels[:samples.shape[0]] = 1
        self.labels[samples.shape[0]:] = 0

        return

    def expand_adjacency_matrix(self, ref_matrix:np.ndarray):
        
        self.expanded_matrix = np.zeros((self.qtd_samples, self.qtd_samples))
    
        # Conectar cada vértice aos seus vizinhos e aos vizinhos dos vizinhos
        for i in range(self.qtd_samples):
            for j in range(self.qtd_samples):
                if ref_matrix[i, j] != 0:
                    self.expanded_matrix[i, j] = 1  # Conectar vértice aos vizinhos
                    for k in range(self.qtd_samples):
                        if ref_matrix[j, k] != 0 and k != i:
                            self.expanded_matrix[i, k] = 1  # Conectar vértice aos vizinhos dos vizinhos


    def label_connected_components(self):
        visited = [False] * self.qtd_samples
        labels = [-1] * self.qtd_samples  # Inicialmente, todos os vértices têm rótulo -1
        current_label = 0

        for vertex in range(self.qtd_samples):
            if not visited[vertex]:
                # Começar uma nova busca em largura a partir do vértice não visitado
                bfs_queue = deque([vertex])
                visited[vertex] = True

                while bfs_queue:
                    current_vertex = bfs_queue.popleft()
                    labels[current_vertex] = current_label

                    # Encontrar vizinhos não visitados e adicioná-los à fila
                    for neighbor in range(self.qtd_samples):
                        if self.expanded_matrix[current_vertex][neighbor] == 1 and not visited[neighbor]:
                            bfs_queue.append(neighbor)
                            visited[neighbor] = True

                current_label += 1  # Atualizar o rótulo para o próximo conjunto de pontos interconectados

        self.labels = labels


    def fit(self, samples: np.ndarray, min_distance: float = 5):
        self.atribute_all_samples_and_labels(samples=samples)

        self.dist_matrix = np.zeros((self.qtd_samples,self.qtd_samples))

        self.model = fema_classifier.FEMaClassifier(k=10,basis=fema_classifier.Basis.shepardBasis)
        self.model.fit(self.all_samples,self.labels.reshape((len(self.labels),1)))

        return
    
    def predict(self, test: np.ndarray, th_same_cluster: float = 0.9, qtd_diff_samples: float = 50):
        
        self.qtd_diff_samples = qtd_diff_samples
        self.th_same_cluster = th_same_cluster

        self.dist_matrix = np.zeros((self.qtd_samples,self.qtd_samples))

        for i in range(self.qtd_samples):
            for j in range(self.qtd_samples):
                if i == j:
                    self.dist_matrix[i,i] = 1
                    continue
                if i > j:
                    continue
                diff = (self.samples[i] - self.samples[j])/(self.qtd_diff_samples-1)
                test_samples = np.zeros((self.qtd_diff_samples,2))
                for k in range(self.qtd_diff_samples):
                    test_samples[k] = (0.999*self.samples[i] - diff*k)
                pred, prob = self.model.predict(test_samples,3)
                if len(pred[prob[:,1] < th_same_cluster]) > 0:
                    self.dist_matrix[i,j] = 0
                    self.dist_matrix[j,i] = 0
                else:
                    self.dist_matrix[i,j] = 1
                    self.dist_matrix[j,i] = 1

        # Expandir a matriz de adjacência
        expanded_matrix = self.expand_adjacency_matrix(self.dist_matrix)


        for i in range(10):
            self.expanded_matrix = self.expand_adjacency_matrix(self.expanded_matrix)

        self.labels = self.label_connected_components()

        return self.labels

"""
def main():
    clustering = FEMaClustering(z=2)
    
    N = 20
    dimensions = 2
    points = clustering.generate_random_points(N, dimensions)
    clustering.plot_points(points)

    clustering.calculate_matrices(points)
    clustering.determine_conquest(points)
    clustering.label_clusters(points)

if __name__ == "__main__":
    main()
"""