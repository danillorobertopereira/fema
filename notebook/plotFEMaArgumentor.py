
import sys
import os
from typing import Tuple

sys.path.append('/home/danillorp/Área de Trabalho/github/fema/src/')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import fema_classifier
import fema_augmentor

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Função para visualizar imagens
def plot_images(data, labels, title):
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    print(labels)
    print(labels.shape)
    print(labels[0])
    for i, ax in enumerate(axes):
        print(i)
        print(labels[i])
        ax.imshow(data[i].reshape(28, 28),cmap='coolwarm')
        ax.set_title(f'{labels[i]}')
        ax.axis('off')
    fig.suptitle(title)
    plt.show()
    
# Função para carregar o dataset MNIST
def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    data = mnist.data
    target = mnist.target
    return data, target

# Função principal
def main():
    # Carregar dados MNIST
    data, target = load_mnist_data()


    # Dividir em treino e teste
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.1, stratify=target)

    # Padronizar os dados
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    train_y = train_y.astype(int)
    train_y = np.array(train_y)
    print(type(train_y))
    # Exibir algumas imagens originais antes da augmentação
    print("Exemplos originais:")
    plot_images(train_x[:10], train_y[:10], title="Original MNIST Digits")


    
    # Identificar as classes com desbalanceamento superior a 25%
    unique_classes, counts = np.unique(train_y, return_counts=True)
    max_count = counts.max()
    #target_classes = unique_classes[counts / max_count < 0.75]
    target_classes = unique_classes[2:5]  # Ajuste para considerar todas as classes

    # Aplicar augmentação se houver classes desbalanceadas
    if len(target_classes) > 0:
        print(f"Classes desbalanceadas detectadas: {target_classes}")

        augmentor = fema_augmentor.FEMaAugmentor(
                k=0, basis=fema_classifier.Basis.shepardBasis, th=0.15, target_classes=target_classes,
                scale=0.25, loc=0.0)
        augmentor.fit(train_x, train_y.reshape((train_y.shape[0], 1)))
        new_samples, new_labels = augmentor.augment(N=int(200 * len(target_classes)))

        # Exibir as imagens geradas
        print("Exemplos gerados pelo FEMaAugmentor:")
        print(len(new_labels))
        plot_images(new_samples, new_labels.ravel(), title="Augmented MNIST Digits")

        # Combinar os dados aumentados com os dados de treino originais
        train_x = np.vstack([train_x, new_samples])
        train_y = np.hstack([train_y, new_labels.ravel()])
    else:
        print("Nenhuma classe desbalanceada detectada para augmentação.")

    
if __name__ == "__main__":
    main()
