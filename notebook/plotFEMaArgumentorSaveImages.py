
import sys
import os
from typing import Tuple

sys.path.append('/home/danillorp/Área de Trabalho/github/fema/src/')


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import fema_classifier
import fema_augmentor

sys.path.append('/home/danillorp/Área de Trabalho/github/fema/src/')

# Função para carregar o dataset MNIST
def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    data = mnist.data
    target = mnist.target
    return data, target

# Função para salvar as imagens geradas
def save_images(data, labels, param_combo, base_dir="augmented_images"):
    for i, (image, label) in enumerate(zip(data, labels)):
        class_dir = os.path.join(base_dir, f"{param_combo}_class_{label}")
        os.makedirs(class_dir, exist_ok=True)
        image_path = os.path.join(class_dir, f"augmented_{i}.png")
        plt.imsave(image_path, image.reshape(28, 28), cmap='gray')

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

    # Identificar as classes para a augmentação
    unique_classes, counts = np.unique(train_y, return_counts=True)
    target_classes = unique_classes  # Considerar todas as classes

    # Parâmetros a serem avaliados
    th_values = [0.1, 0.15, 0.2]
    scale_values = [0.1, 0.25, 0.5]
    
    # Dicionário para armazenar resultados
    results = {}

    # Avaliar impacto de cada combinação de parâmetros
    for th in th_values:
        for scale in scale_values:
            param_combo = f"th_{th}_scale_{scale}"
            print(f"Avaliando combinação: {param_combo}")
            
            augmentor = fema_augmentor.FEMaAugmentor(
                    k=0, basis=fema_classifier.Basis.shepardBasis, th=th, target_classes=target_classes,
                    scale=scale, loc=0.0)
            augmentor.fit(train_x, train_y.reshape((train_y.shape[0], 1)))
            new_samples, new_labels = augmentor.augment(N=int(200 * len(target_classes)))

            # Salvar as imagens geradas
            save_images(new_samples, new_labels.ravel(), param_combo=param_combo)
            
            # Combinar os dados aumentados com os dados de treino originais
            augmented_train_x = np.vstack([train_x, new_samples])
            augmented_train_y = np.hstack([train_y, new_labels.ravel()])
            
            # Treinar um classificador simples com os dados aumentados
            clf = LogisticRegression(max_iter=1000)
            clf.fit(augmented_train_x, augmented_train_y)
            
            # Avaliar a acurácia no conjunto de teste
            predictions = clf.predict(test_x)
            accuracy = accuracy_score(test_y, predictions)
            results[param_combo] = accuracy
            print(f"Acurácia com {param_combo}: {accuracy:.4f}")

    # Exibir os resultados
    print("\nResultados finais:")
    for param_combo, accuracy in results.items():
        print(f"{param_combo}: {accuracy:.4f}")

if __name__ == "__main__":
    main()
