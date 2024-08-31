import sys
import os
from typing import Tuple
from fem_basis import Basis  
import numpy as np
from sklearn.neighbors import NearestNeighbors


class FEMaAugmentor:
    def __init__(self, k: int = 2, basis=Basis.shepardBasis, th: float = 0.5, loc: float = 0.0, scale: int = 0.15, z: int = 3, target_classes=None, use_smart_sampling: bool = False) -> None:
        self.k = k
        self.z = z
        self.basis = basis
        self.th = th
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

                if label_new == y_base and probabilities[label_new] > self.th:
                    new_samples.append(x_new)
                    new_labels.append(label_new)

        return np.array(new_samples), np.array(new_labels).reshape(-1, 1)

    def _generate_sample(self, x_base: np.array) -> np.array:
        perturbation = np.random.normal(loc=self.loc, scale=self.scale, size=self.num_features)
        return x_base + perturbation

    def _generate_smart_sample(self, x_base: np.array, y_base: int) -> np.array:
        k = self.k
        minority_idx = np.where(self.train_y[:, 0] == y_base)[0]

        if self.k <= 1:
            k = len(self.train_x[minority_idx])
        
        # Encontrar vizinhos da mesma classe
        neighbors = NearestNeighbors(n_neighbors=k).fit(self.train_x[minority_idx])
        neighbor_idx = neighbors.kneighbors([x_base], return_distance=False).ravel()
        random_neighbor = self.train_x[minority_idx[neighbor_idx[np.random.randint(1, k)]]]

        # Gerar uma amostra sintética entre x_base e o vizinho selecionado
        alpha = np.random.uniform(0, 1)
        synthetic_sample = x_base + alpha * (random_neighbor - x_base)

        # Amostragem em regiões de vazios
        # Gerar uma amostra aleatória dentro do espaço das características
        empty_region_sample = np.random.uniform(np.min(self.train_x, axis=0), np.max(self.train_x, axis=0))
        
        # Otimização: maximizar a distância da amostra sintética para os pontos de outras classes
        other_classes_idx = np.where(self.train_y[:, 0] != y_base)[0]
        other_class_samples = self.train_x[other_classes_idx]

        distances = np.linalg.norm(other_class_samples - synthetic_sample, axis=1)
        max_distance_idx = np.argmax(distances)
        farthest_other_class_sample = other_class_samples[max_distance_idx]

        # Ajuste final: interpolar entre a amostra sintética e o ponto de outra classe mais distante
        beta = np.random.uniform(0.4, 0.6)
        final_sample = beta * synthetic_sample + (1 - beta) * farthest_other_class_sample

        # Verificar se a amostra gerada está na região de vazio e usar uma amostra vazia se estiver
        if np.linalg.norm(empty_region_sample - x_base) > np.linalg.norm(final_sample - x_base):
            return empty_region_sample
        else:
            return final_sample
