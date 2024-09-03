import sys
import os
from typing import Tuple
from fem_basis import Basis  
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from imblearn.under_sampling import EditedNearestNeighbours

class FEMaAugmentor:
    def __init__(self, k: int = 0, basis=Basis.shepardBasis, th_min: float = 0.3, th_max: float = 0.7, loc: float = 0.0, scale: int = 0.15, z: int = 3, 
                 target_classes=None, use_smart_sampling: bool = False, apply_enn: bool = False) -> None:
        self.k = k
        self.z = z
        self.basis = basis
        self.th_min = th_min
        self.th_max = th_max
        self.loc = loc
        self.scale = scale
        self.target_classes = target_classes
        self.train_x = None
        self.train_y = None
        self.num_train_samples = 0
        self.num_features = 0
        self.num_classes = 0
        self.probability_classes = None
        self.use_smart_sampling = use_smart_sampling
        self.apply_enn = apply_enn

    def fit(self, train_x: np.array, train_y: np.array) -> None:
        self.train_x = train_x
        self.train_y = train_y
        self.num_train_samples = len(train_y)
        self.num_features = self.train_x.shape[1]
        self.num_classes = len(np.unique(train_y[:, 0]))

        self.probability_classes = np.zeros((self.num_classes, self.num_train_samples))

        for i in range(self.num_classes):
            self.probability_classes[i, :] = (train_y[:, 0] == i).astype(float)

    def augment(self, N: int) -> Tuple[np.array, np.array]:
        new_samples = []
        new_labels = []

        for _ in range(N):
            idx = np.random.randint(0, self.num_train_samples)
            x_base = self.train_x[idx]
            y_base = self.train_y[idx, 0]

            if self.target_classes is None or y_base in self.target_classes:
                if self.use_smart_sampling:
                    x_new = self._generate_smart_sample(x_base, y_base)
                else:
                    x_new = self._generate_sample(x_base)

                probabilities = [self.basis(train_x=self.train_x, train_y=self.probability_classes[c], test_one_sample=x_new, k=self.k, z=3) for c in range(self.num_classes)]
                label_new = np.argmax(probabilities)
                max_probability = probabilities[label_new]

                # Verifica se a probabilidade está dentro dos limites
                if label_new == y_base and self.th_min <= max_probability <= self.th_max:
                    new_samples.append(x_new)
                    new_labels.append(label_new)

        new_samples = np.array(new_samples)
        new_labels = np.array(new_labels).reshape(-1, 1)

        if self.apply_enn:
            new_samples, new_labels = self._apply_enn(new_samples, new_labels)

        return new_samples, new_labels

    def _generate_sample(self, x_base: np.array) -> np.array:
        perturbation = np.random.normal(loc=self.loc, scale=self.scale, size=self.num_features)
        return x_base + perturbation

    def _generate_smart_sample(self, x_base: np.array, y_base: int) -> np.array:
        # Encontra os pontos da classe minoritária
        minority_class_points = self.train_x[self.train_y[:, 0] == y_base]
        
        # Encontra os pontos das outras classes
        other_class_points = self.train_x[self.train_y[:, 0] != y_base]

        # Calcula as distâncias entre cada ponto da classe minoritária e os pontos das outras classes
        distances = cdist(minority_class_points, other_class_points, metric='euclidean')
        
        # Para cada ponto da classe minoritária, encontra o ponto mais próximo de outra classe
        nearest_distances = np.min(distances, axis=1)
        nearest_indices = np.argmin(distances, axis=1)
        nearest_points = other_class_points[nearest_indices]
        
        # Gera novos pontos aleatoriamente dentro do raio da distância ao ponto mais próximo de outra classe
        random_radii = np.random.uniform(low=0.0, high=1.0, size=nearest_distances.shape) * nearest_distances
        random_directions = np.random.normal(size=nearest_points.shape)
        random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)  # Normaliza os vetores

        # Calcula os novos pontos como deslocamentos dentro do raio do ponto mais próximo de outra classe
        new_samples = minority_class_points + random_radii[:, np.newaxis] * random_directions
        
        # Adiciona um ruído gaussiano aos novos pontos
        noise = np.random.normal(loc=self.loc, scale=self.scale, size=new_samples.shape)
        new_samples += noise

        # Retorna um dos novos pontos gerados
        idx = np.random.randint(0, new_samples.shape[0])
        return new_samples[idx]


    def _apply_enn(self, samples: np.array, labels: np.array) -> Tuple[np.array, np.array]:
        try:
            enn = EditedNearestNeighbours(n_neighbors=max(1, self.k))
            samples_resampled, labels_resampled = enn.fit_resample(samples, labels.ravel())
        except Exception as e:
            print(f"Error applying ENN: {e}")
            samples_resampled, labels_resampled = samples, labels
        return samples_resampled, labels_resampled.reshape(-1, 1)
