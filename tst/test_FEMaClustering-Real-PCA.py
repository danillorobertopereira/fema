import sys
import os
from typing import Tuple

sys.path.append('/home/danillorp/Área de Trabalho/github/fema/src/')


from fema_clustering import FEMaClustering
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def main():
    clustering = FEMaClustering(z=2)
    
    N = 100
    dimensions = 2
    points = clustering.generate_random_points(bounds=[(0,200),(0,200)],num_points=N)
    #clustering.plot_points(points)

    clustering.fit(points)
    
    # Plotar pontos
    if dimensions == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(clustering.samples[:,0], clustering.samples[:,1], color='blue', label='Nuvem de Pontos Inicial (NP)')
        plt.scatter(clustering.random_samples[:,0], clustering.random_samples[:,1], color='red', label='Pontos Aleatórios')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Nuvem de Pontos Inicial com Pontos Aleatórios')
        plt.grid(True)
        plt.show()
    
    print(clustering.predict(th_same_cluster=0.85))

    if dimensions == 2:
        fig, ax = plt.subplots()

        # Plot the generated points
        ax.scatter(points[:, 0], points[:, 1], c=clustering.labels, marker='o', edgecolors='black')
        ax.grid(True)

        for i, txt in enumerate(range(N)):
            ax.annotate(clustering.labels[i], (points[i,0], points[i,1]))

        plt.show()
        


if __name__ == "__main__":
    main()
