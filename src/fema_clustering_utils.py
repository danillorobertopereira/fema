import sys
import os
from typing import Tuple

from numpy.random.mtrand import random


sys.path.append('C:\\Users\\coton\\Desktop\\github\\fema\\src\\')


import numpy as np
import math 

from fem_basis import Basis    





def generate_random_points(ranges, num_points):
    """
    Generates random points in N dimensions within the specified ranges.

    Args:
        ranges (list of tuple): List of tuples where each tuple contains the range (min, max) for each dimension.
        num_points (int): Number of random points to generate.

    Returns:
        np.ndarray: 2D array where each row represents a point in N dimensions.
    """
    points = [np.random.uniform(low, high, num_points) for low, high in ranges]
    return np.array(points).T

def filter_nearby_points(random_points, np_points, min_distance):
    """
    Filters random points that are at least a minimum distance away from all NP points.

    Args:
        random_points (np.ndarray): 2D array of random points, where each row is a point in N dimensions.
        np_points (np.ndarray): 2D array of NP points, where each row is a point in N dimensions.
        min_distance (float): Minimum distance that each random point must have from all NP points.

    Returns:
        np.ndarray: 2D array of filtered points, where each row is a point in N dimensions.
    """
    filtered_points = []
    
    for random_point in random_points:
        # Calculate the Euclidean distance between the random point and all NP points
        distances = np.linalg.norm(np_points - random_point, axis=1)
        
        # Check if all distances are greater than or equal to the minimum distance
        if np.all(distances >= min_distance):
            filtered_points.append(random_point)
    
    return np.array(filtered_points)

def lists_to_np_array(lists):
    """
    Converts a list of lists to a NumPy array, where each inner list represents a dimension.

    Args:
        lists (list of list): List of lists where each inner list contains the values of one dimension.

    Returns:
        np.ndarray: 2D array where each column represents a dimension.
    """
    # Convert each list to a NumPy array
    arrays = [np.array(lst) for lst in lists]
    
    # Stack the arrays into a 2D array where each column is a dimension
    coordinates = np.column_stack(arrays)
    
    return coordinates




# Example usage:
# Defining ranges for each dimension
ranges = [(0, 10), (0, 10), (0, 10)]
num_points = 100

# Generating random points in 3 dimensions
random_points = generate_random_points(ranges, num_points)

# Defining example NP points
np_points = np.array([[2, 2, 2], [8, 8, 8]])

# Filtering random points that are at least 2 units away from NP points
min_distance = 2
filtered_points = filter_nearby_points(random_points, np_points, min_distance)

print("Generated random points:", random_points)
print("Filtered points:", filtered_points)

#Putting Labels
samples = np_points.copy()
non_samples = lists_to_np_array(filtered_points)

print(samples.shape,non_samples.shape,len(filtered_points))

joined_samples = np.concatenate((samples, non_samples))

print(joined_samples.shape)

labels = np.zeros(joined_samples.shape[0],dtype=int)
labels[:samples.shape[0]] = 1
labels[samples.shape[0]:] = 0
print(labels)
print(labels.shape)

# Definindo os limites do espaço
x_min, x_max = 0, 100
y_min, y_max = 0, 100

# Definindo o passo
step = 2

# Gerando os pontos usando np.meshgrid
x_values = np.arange(x_min, x_max + step, step)
y_values = np.arange(y_min, y_max + step, step)

xx, yy = np.meshgrid(x_values, y_values)

# Convertendo os pontos gerados em um único conjunto de pontos 2D
test = np.column_stack((xx.ravel(), yy.ravel()))

print("Primeiros 10 pontos:")
print(points[:10])

# Plotando os pontos (opcional)
plt.figure(figsize=(8, 8))
plt.scatter(test[:, 0], test[:, 1],  c='blue')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Conjunto de Pontos 2D')
plt.grid(True)
plt.show()

model = fema_classifier.FEMaClassifier(k=10,basis=fema_regression.Basis.shepardBasis)
model.fit(joined_samples,labels.reshape((len(labels),1)))



pred, prob = model.predict(test,3)


