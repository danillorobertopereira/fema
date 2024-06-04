import sys
import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.datasets import fetch_openml, fetch_20newsgroups
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


sys.path.append('/home/danillorp/Área de Trabalho/github/fema/src/')

from fema_clustering import FEMaClustering
import matplotlib.pyplot as plt

# Função para carregar e preparar os datasets

def load_datasets():
    # Carregar datasets do sklearn
    print('Loading dataset Iris ...')
    iris = datasets.load_iris()
    print('Loading dataset Wine ...')
    wine = datasets.load_wine()
    print('Loading dataset Digits ...')
    digits = datasets.load_digits()
    
    # Carregar MNIST
    
    """print('Loading dataset MNIST ...')
    mnist = fetch_openml('mnist_784', version=1)
    mnist_data = mnist.data
    mnist_target = mnist.target.astype(int)
    
    # Carregar KDD Cup 99
    print('Loading dataset KDDCup99 ...')
    kddcup = fetch_openml('KDDCup99', version=1)
    kddcup_data = kddcup.data
    kddcup_target = kddcup.target

    # Carregar CIFAR-10
    print('Loading dataset CIFAR-10 ...')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    cifar10 = CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_data = torch.stack([data for data, _ in DataLoader(cifar10, batch_size=len(cifar10))]).squeeze()
    cifar10_target = np.array(cifar10.targets)
    
    # Carregar CIFAR-100
    print('Loading dataset CIFAR-100 ...')
    cifar100 = CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar100_data = torch.stack([data for data, _ in DataLoader(cifar100, batch_size=len(cifar100))]).squeeze()
    cifar100_target = np.array(cifar100.targets)

    # Carregar 20 Newsgroups
    print('Loading dataset 20newsgroups ...')
    newsgroups = fetch_20newsgroups(subset='all')
    newsgroups_data = newsgroups.data
    newsgroups_target = newsgroups.target

    return {
        'Iris': (iris.data, iris.target),
        'Wine': (wine.data, wine.target),
        'Digits': (digits.data, digits.target),
        'MNIST': (mnist_data, mnist_target),
        'KDD Cup 99': (kddcup_data, kddcup_target),
        'CIFAR-10': (cifar10_data, cifar10_target),
        'CIFAR-100': (cifar100_data, cifar100_target),
        '20 Newsgroups': (newsgroups_data, newsgroups_target)
    }
    
    """
    return {
        'Iris': (iris.data, iris.target),
        'Wine': (wine.data, wine.target),
      }
     
# Função para normalizar e reduzir dimensionalidade
def preprocess_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data_scaled)
    return data_reduced

# Função para aplicar métodos de clusterização
def apply_clustering_methods(data):
    clustering_methods = {
        'KMeans': KMeans(n_clusters=3, random_state=42),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'Agglomerative': AgglomerativeClustering(n_clusters=3),
        'GMM': GaussianMixture(n_components=3, random_state=42),
        'Spectral': SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42),
        'FEMaClustering': FEMaClustering(z=2)
    }
    
    clusters = {}
    for method_name, method in clustering_methods.items():
        if method_name == 'GMM':
            method.fit(data)
            labels = method.predict(data)
        elif method_name == 'FEMaClustering':
            print('FEMaClustering... ...')
            method.fit(data,min_distance=0.1,qtd_samples_perc=0.25)
            labels = method.predict(th_same_cluster=0.9,qtd_diff_samples=50)
        else:
            method.fit(data)
            labels = method.labels_
        clusters[method_name] = labels
        
    return clusters

# Função para calcular as métricas
def calculate_metrics(data, labels_true, labels_pred):
    metrics = {
        'Silhouette Score': silhouette_score(data, labels_pred),
        'Davies-Bouldin Score': davies_bouldin_score(data, labels_pred),
        'Adjusted Rand Index': adjusted_rand_score(labels_true, labels_pred),
        'Normalized Mutual Information': normalized_mutual_info_score(labels_true, labels_pred)
    }
    return metrics


# Supondo que as funções load_datasets, preprocess_data, apply_clustering_methods e calculate_metrics já existam

preprocess_data_flag = False
N = 10  # Número de repetições

# Função principal para executar o experimento
def main():
    datasets = load_datasets()
    
    all_results = []

    for repetition in range(N):
        print(f"\nRepetition {repetition + 1}/{N}")

        for dataset_name, (data, target) in datasets.items():
            print(f"\nProcessing {dataset_name} dataset")
            print('INFO:', data.shape)
            
            if dataset_name in ['20 Newsgroups']:
                # Para o conjunto de dados 20 Newsgroups, a vetorização do texto é necessária
                vectorizer = TfidfVectorizer(max_features=1000)
                data = vectorizer.fit_transform(data).toarray()
            
            if preprocess_data_flag:
                data_preprocessed = preprocess_data(data)
            else:
                data_preprocessed = data.copy()

            clusters = apply_clustering_methods(data_preprocessed)
            
            for method_name, labels_pred in clusters.items():
                metrics = calculate_metrics(data_preprocessed, target, labels_pred)
                
                # Adiciona os resultados para esta repetição e método na lista de resultados
                result_row = {
                    'Repetition': repetition + 1,
                    'Dataset': dataset_name,
                    'Method': method_name
                }
                result_row.update(metrics)
                all_results.append(result_row)

                print(f"\nResults for {method_name} on {dataset_name}:")
                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value:.4f}")

                if preprocess_data_flag:
                    plt.figure()  # Cria uma nova figura para cada método
                    plt.scatter(data_preprocessed[:, 0], data_preprocessed[:, 1], c=labels_pred)  # Plotar dados com cores de cluster
                    plt.title(f"{method_name} on {dataset_name}")
                    plt.xlabel("Component 1")
                    plt.ylabel("Component 2")
                    plt.savefig(f"figs/{dataset_name}_{method_name}_rep{repetition + 1}.png")

    # Salva todos os resultados em um arquivo CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('clustering_results.csv', index=False)

if __name__ == "__main__":
    main()


"""Para o Silhouette Score e Davies-Bouldin Score, quanto maior o valor, melhor é o agrupamento.
Para o ARI e NMI, quanto mais próximo de 1, melhor é a concordância entre os agrupamentos e os rótulos verdadeiros."""
