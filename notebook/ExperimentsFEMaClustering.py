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
import time
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification

#from pyinstrument import Profiler

sys.path.append('/home/danillorp/Área de Trabalho/github/fema/src/')

from fema_clustering import FEMaClustering
import matplotlib.pyplot as plt

# Função para carregar e preparar os datasets

def load_datasets():
    # Função para filtrar datasets com base nos critérios
    def filter_dataset(data, target, max_samples=1000000, max_features=30000):
        if data.shape[0] <= max_samples and data.shape[1] <= max_features:
            return data, target
        return None, None

    datasets_dict = {}

    # Adicionando novos datasets
    print('Loading dataset Breast Cancer Wisconsin (Diagnostic) ...')
    bc_wisconsin = datasets.load_breast_cancer()
    data, target = filter_dataset(bc_wisconsin.data, bc_wisconsin.target)
    if data is not None:
        datasets_dict['Breast Cancer Wisconsin (Diagnostic)'] = (data, target)
        print(data.shape[0],data.shape[1])
    
    print('Loading dataset Diabetes ...')
    diabetes = datasets.load_diabetes()
    data, target = filter_dataset(diabetes.data, diabetes.target)
    if data is not None:
        datasets_dict['Diabetes'] = (data, target)
        print(data.shape[0],data.shape[1])
    #kc1 1067
    #pc1 1068
    #dna 40670
    #churn 40701
    
    print('Loading dataset kc1 ...')
    kc1 = fetch_openml(data_id=1067)
    data, target = filter_dataset(kc1.data, kc1.target)
    if data is not None:
        datasets_dict['kc1'] = (data, target)
        print(data.shape[0],data.shape[1])

    print('Loading dataset DNA ...')
    dna = fetch_openml(data_id=40670)
    data, target = filter_dataset(dna.data, dna.target)
    if data is not None:
        datasets_dict['dna'] = (data, target)
        print(data.shape[0],data.shape[1])

    print('Loading dataset Churn ...')
    churn = fetch_openml(data_id=40701)
    data, target = filter_dataset(churn.data, churn.target)
    if data is not None:
        datasets_dict['churn'] = (data, target)
        print(data.shape[0],data.shape[1])

    """
    print('Loading dataset Satelite ...')
    satelite = fetch_openml(data_id=40900)
    data, target = filter_dataset(satelite.data, satelite.target)
    if data is not None:
        datasets_dict['Satelite'] = (data, target)
        print(data.shape[0],data.shape[1])

    print('Loading dataset One-Hundred ...')
    oneh = fetch_openml(data_id=1493)
    data, target = filter_dataset(oneh.data, oneh.target)
    if data is not None:
        datasets_dict['One-Hundred'] = (data, target)
        print(data.shape[0],data.shape[1])
    """
        
    # Carregar datasets do sklearn
    print('Loading dataset Iris ...')
    iris = datasets.load_iris()
    data, target = filter_dataset(iris.data, iris.target)
    if data is not None:
        datasets_dict['Iris'] = (data, target)
        print(data.shape[0],data.shape[1])

    print('Loading dataset Wine ...')
    wine = datasets.load_wine()
    data, target = filter_dataset(wine.data, wine.target)
    if data is not None:
        datasets_dict['Wine'] = (data, target)
        print(data.shape[0],data.shape[1])
    
    print('Loading dataset California Housing ...')
    cal_housing = fetch_openml(name='California', version=1, as_frame=False)
    data, target = filter_dataset(cal_housing.data, cal_housing.target)
    if data is not None:
        datasets_dict['California Housing'] = (data, target)
        print(data.shape[0],data.shape[1])
    
    # Adicionar dataset com aproximadamente 20.000 amostras
    """print('Loading dataset Fashion MNIST (20,000 samples) ...')
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    data, target = fashion_mnist.data, fashion_mnist.target.astype(int)
    if data.shape[0] > 20000:
        indices = np.random.choice(data.shape[0], 20000, replace=False)
        data, target = data[indices], target[indices]
    datasets_dict['Fashion MNIST (20,000 samples)'] = (data, target)
    """
    # Carregar o dataset Digits por último
    print('Loading dataset Digits ...')
    digits = datasets.load_digits()
    data, target = filter_dataset(digits.data, digits.target)
    if data is not None:
        datasets_dict['Digits'] = (data, target)
        print(data.shape[0],data.shape[1])

    return datasets_dict



def generate_toy_datasets():
    datasets = {}

    n_samples = 500
    # Dataset 1: Blobs
    print('Creating Blobs dataset ...')
    X_blobs, y_blobs = make_blobs(n_samples=n_samples, centers=4, cluster_std=0.60)
    datasets['Blobs'] = (X_blobs, y_blobs)

    # Dataset 2: Noisy Circles
    print('Creating Noisy Circles dataset ...')
    X_circles, y_circles = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    datasets['Noisy Circles'] = (X_circles, y_circles)

    # Dataset 3: Noisy Moons
    print('Creating Noisy Moons dataset ...')
    X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.1)
    datasets['Noisy Moons'] = (X_moons, y_moons)
    
    # Dataset 4: Anisotropicly Distributed Blobs
    print('Creating Anisotropic Blobs dataset ...')
    X_aniso, y_aniso = make_blobs(n_samples=n_samples)
    transformation = [[0.3, -0.3], [-0.2, 0.8]]
    X_aniso = np.dot(X_aniso, transformation)
    datasets['Anisotropic Blobs'] = (X_aniso, y_aniso)

    # Dataset 5: Varied Blobs
    print('Creating Varied Blobs dataset ...')
    X_varied, y_varied = make_blobs(n_samples=n_samples, centers=4, cluster_std=[1.0, 2.5, 0.5, 1.5])
    datasets['Varied Blobs'] = (X_varied, y_varied)

    # Dataset 6: Gaussian Quantiles
    print('Creating Gaussian Quantiles dataset ...')
    X_quantiles, y_quantiles = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
    datasets['Gaussian Quantiles'] = (X_quantiles, y_quantiles)

    # Dataset 7: No Structure
    print('Creating No Structure dataset ...')
    X_no_structure = np.random.rand(150, 2)
    y_no_structure = np.zeros(1500)
    datasets['No Structure'] = (X_no_structure, y_no_structure)

    return datasets

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
        'KMeans': KMeans(n_clusters=3),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'Agglomerative': AgglomerativeClustering(n_clusters=3),
        'GMM': GaussianMixture(n_components=3),
        'Spectral': SpectralClustering(n_clusters=3, affinity='nearest_neighbors'),
        'FEMaClustering-0.99': FEMaClustering(z=2),
        'FEMaClustering-0.97': FEMaClustering(z=2),
        'FEMaClustering-0.95': FEMaClustering(z=2),
        'FEMaClustering-0.90': FEMaClustering(z=2),
        'FEMaClustering-0.85': FEMaClustering(z=2),
        'FEMaClustering-0.80': FEMaClustering(z=2),
        'FEMaClustering-0.75': FEMaClustering(z=2),
        'FEMaClustering-0.70': FEMaClustering(z=2),
        'FEMaClustering-0.65': FEMaClustering(z=2),
        'FEMaClustering-0.60': FEMaClustering(z=2),
        'FEMaClustering-0.55': FEMaClustering(z=2),
        'FEMaClustering-0.50': FEMaClustering(z=2),
        
    }
        
    
    clusters = {}
    time_exec = {}
    fema_trained = False
    model_trained = None
    for method_name, method in clustering_methods.items():
       
        start_time = time.time()     


        if method_name.startswith('FEMaClustering') and fema_trained == False:
            print('FEMaClustering... ...')
            method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            fema_trained = True
            model_trained = method

        if method_name == 'GMM':
            method.fit(data)
            labels = method.predict(data)
        elif method_name == 'FEMaClustering':
            print('FEMaClustering-0.95... ...')
            #method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            labels = model_trained.predict(th_same_cluster=0.95,qtd_diff_samples=20)
        elif method_name == 'FEMaClustering-0.99':
            print('FEMaClustering-0.99... ...')
            #method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            labels = model_trained.predict(th_same_cluster=0.99,qtd_diff_samples=20)
        elif method_name == 'FEMaClustering-0.97':
            print('FEMaClustering-0.97... ...')
            #method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            labels = model_trained.predict(th_same_cluster=0.975,qtd_diff_samples=20)
        elif method_name == 'FEMaClustering-0.95':
            print('FEMaClustering-0.95... ...')
            #method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            labels = model_trained.predict(th_same_cluster=0.95,qtd_diff_samples=20)
        elif method_name == 'FEMaClustering-0.90':
            print('FEMaClustering-0.90... ...')
            #method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            labels = model_trained.predict(th_same_cluster=0.90,qtd_diff_samples=20)
        elif method_name == 'FEMaClustering-0.85':
            print('FEMaClustering-0.85... ...')
            #method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            labels = model_trained.predict(th_same_cluster=0.85,qtd_diff_samples=20)
        elif method_name == 'FEMaClustering-0.80':
            print('FEMaClustering-0.80... ...')
            #method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            labels = model_trained.predict(th_same_cluster=0.80,qtd_diff_samples=20)
        elif method_name == 'FEMaClustering-0.75':
            print('FEMaClustering-0.75... ...')
            #method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            labels = model_trained.predict(th_same_cluster=0.75,qtd_diff_samples=20)
        elif method_name == 'FEMaClustering-0.70':
            print('FEMaClustering-0.70... ...')
            #method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            labels = model_trained.predict(th_same_cluster=0.70,qtd_diff_samples=20)
        elif method_name == 'FEMaClustering-0.65':
            print('FEMaClustering-0.65... ...')
            #method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            labels = model_trained.predict(th_same_cluster=0.65,qtd_diff_samples=20)
        elif method_name == 'FEMaClustering-0.60':
            print('FEMaClustering-0.60... ...')
            #method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            labels = model_trained.predict(th_same_cluster=0.60,qtd_diff_samples=20)
        elif method_name == 'FEMaClustering-0.55':
            print('FEMaClustering-0.55... ...')
            #method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            labels = model_trained.predict(th_same_cluster=0.55,qtd_diff_samples=20)
        elif method_name == 'FEMaClustering-0.50':
            print('FEMaClustering-0.50... ...')
            #method.fit(data,min_distance=0.2,qtd_samples_perc=0.99)
            labels = model_trained.predict(th_same_cluster=0.50,qtd_diff_samples=20)
        else:
            method.fit(data)
            labels = method.labels_
        
        end_time = time.time()  
        processing_time = end_time - start_time 
        clusters[method_name] = labels
        time_exec[method_name] = processing_time
    
    return clusters, time_exec

# Função para calcular as métricas
def calculate_metrics(data, labels_true, labels_pred):
    try:
        metrics = {
            'Silhouette Score': silhouette_score(data, labels_pred),
            'Davies-Bouldin Score': davies_bouldin_score(data, labels_pred),
            'Adjusted Rand Index': adjusted_rand_score(labels_true, labels_pred),
            'Normalized Mutual Information': normalized_mutual_info_score(labels_true, labels_pred)
        }
    except ValueError:
        metrics = {
            'Silhouette Score': np.nan,
            'Davies-Bouldin Score': np.nan,
            'Adjusted Rand Index': np.nan,
            'Normalized Mutual Information': np.nan
        }
    return metrics


# Supondo que as funções load_datasets, preprocess_data, apply_clustering_methods e calculate_metrics já existam

preprocess_data_flag = False
N = 10  # Número de repetições
is_toy = False

# Função principal para executar o experimento
def main():
    if is_toy:
        datasets = generate_toy_datasets()
    else:
        datasets = load_datasets()
    
    all_results = []
    
 #   profiler = Profiler()
 #   profiler.start()
    for repetition in range(N):
        print("\nRepetition {repetition + 1}/{N}")

        if is_toy:
            datasets = generate_toy_datasets()

        for dataset_name, (data, target) in datasets.items():
            print("\nProcessing ",dataset_name," dataset")
            print('INFO:', data.shape)
            
            if dataset_name in ['20 Newsgroups']:
                # Para o conjunto de dados 20 Newsgroups, a vetorização do texto é necessária
                vectorizer = TfidfVectorizer(max_features=1000)
                data = vectorizer.fit_transform(data).toarray()
            
            if preprocess_data_flag:
                data_preprocessed = preprocess_data(data)
            else:
                data_preprocessed = data.copy()

            clusters, time_exec = apply_clustering_methods(data_preprocessed)
            
            
            for method_name, labels_pred in clusters.items():
                metrics = calculate_metrics(data_preprocessed, target, labels_pred)
                
                # Adiciona os resultados para esta repetição e método na lista de resultados
                result_row = {
                    'Repetition': repetition + 1,
                    'Dataset': dataset_name,
                    'Method': method_name,
                    'Time':  time_exec[method_name]
                }
                result_row.update(metrics)
                all_results.append(result_row)

                print("\nResults for ",method_name," on ",dataset_name)
                for metric_name, metric_value in metrics.items():
                    print(metric_name," ",metric_value)

                if preprocess_data_flag or is_toy:
                    plt.figure()  # Cria uma nova figura para cada método
                    plt.scatter(data_preprocessed[:, 0], data_preprocessed[:, 1], c=labels_pred)  # Plotar dados com cores de cluster
                    plt.title("{method_name} on {dataset_name}")
                    plt.xlabel("Component 1")
                    plt.ylabel("Component 2")
                    plt.savefig(f"figs/"+dataset_name+"_"+"_rep"+str(repetition + 1)+method_name+".png")
                else:
                    print('NO PLOT {dataset_name}_{method_name}')
    # Salva todos os resultados em um arquivo CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('clustering_results.csv', index=False)

#    profiler.stop()
#    profiler.print()


if __name__ == "__main__":
    main()



"""Para o Silhouette Score e Davies-Bouldin Score, quanto maior o valor, melhor é o agrupamento.
Para o ARI e NMI, quanto mais próximo de 1, melhor é a concordância entre os agrupamentos e os rótulos verdadeiros."""