import sys
import os
from typing import Tuple

from numpy.random.mtrand import random


sys.path.append('C:\\Users\\coton\\Desktop\\github\\fema\\src\\')


import numpy as np
import math 

from fem_basis import Basis    
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class FEMaClustering:
    def __init__(self, z: float = 2, samples=None, Basis=Basis.shepardBasis):
        self.z = z
        self.dist_matrix = None
        self.weight_matrix = None
        self.conquest = None
        self.conquested = None
        self.splited = None
        self.samples = samples
        self.basis = Basis(z)

    def generate_random_points(self, n_points: int, dimensions: int = 2) -> np.ndarray:
        return np.random.rand(n_points, dimensions) * 100

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

    def fit(self, X: np.ndarray, y: np.ndarray = None):
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