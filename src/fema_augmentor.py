import numpy as np
from typing import Tuple
from fem_basis import Basis  

class FEMaAugmentor:
    def __init__(self, k: int = 2, basis=Basis.shepardBasis, th: float = 0.5, loc: float = 0.0, scale: int = 0.15, target_classes=None) -> None:
        """Constructor that initializes the FEMa Augmentor

        Args:
            k (int): Number of neighbors used for interpolation
            basis: The finite element basis function
            th (float): Threshold for adding new samples
            target_classes (list or None): List of classes to augment. If None, all classes are augmented.
        """
        self.k = k
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

    def fit(self, train_x: np.array, train_y: np.array) -> None:
        """Fit the FEMa Augmentor with training data

        Args:
            train_x (np.array): Features of the training set
            train_y (np.array): Labels of the training set
        """
        self.train_x = train_x
        self.train_y = train_y
        self.num_train_samples = len(train_y)
        self.num_features = self.train_x.shape[1]
        self.num_classes = len(np.unique(train_y[:, 0]))

        self.probability_classes = np.zeros((self.num_classes, self.num_train_samples))

        for i in range(self.num_classes):
            self.probability_classes[i, :] = (train_y[:, 0] == i).astype(float)

    def augment(self, N: int) -> Tuple[np.array, np.array]:
        """Generate N new samples based on FEMa approach

        Args:
            N (int): Number of new samples to generate

        Returns:
            Tuple[np.array, np.array]: Augmented features and labels
        """
        new_samples = []
        new_labels = []

        for _ in range(N):
            idx = np.random.randint(0, self.num_train_samples)
            x_base = self.train_x[idx]
            y_base = self.train_y[idx, 0]

            if self.target_classes is None or y_base in self.target_classes:
                x_new = self._generate_sample(x_base)

                probabilities = [self.basis(train_x=self.train_x, train_y=self.probability_classes[c], test_one_sample=x_new, k=self.k, z=3) for c in range(self.num_classes)]
                label_new = np.argmax(probabilities)

                if label_new == y_base and probabilities[label_new] > self.th:
                    new_samples.append(x_new)
                    new_labels.append(label_new)

        return np.array(new_samples), np.array(new_labels).reshape(-1, 1)

    def _generate_sample(self, x_base: np.array) -> np.array:
        """Generate a new sample around the given base point using small perturbations

        Args:
            x_base (np.array): Base point to generate new sample around

        Returns:
            np.array: New generated sample
        """
        perturbation = np.random.normal(loc=self.loc, scale=self.scale, size=self.num_features)
        return x_base + perturbation