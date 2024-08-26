
import sys
import os
from typing import Tuple

sys.path.append('/home/danillorp/Área de Trabalho/github/fema/src/')


import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import fema_classifier
import fema_augmentor

from sklearn.utils import class_weight

# Função para tratar features do tipo string
def handle_string_features(data):
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        print(f"Handling string features: {list(string_columns)}")
        for col in string_columns:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_features = encoder.fit_transform(data[[col]])
            encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out([col]))
            # Remover a coluna original de string e adicionar as colunas codificadas
            data = pd.concat([data.drop([col], axis=1), encoded_df], axis=1)
    return data


# Função para avaliar o modelo
def evaluate_model(train_x, train_y, test_x, test_y, classifier, augment=False):
    
    train_y = np.array(train_y)
    train_x = np.array(train_x)
    
    test_y = np.array(test_y)
    test_x = np.array(test_x)

    train_y = train_y.reshape((train_y.shape[0], 1))

    if augment:

        print(f"BEFORE Evaluating dataset")
        print(f"Sample counts per class: {np.bincount(train_y[:,0])}")
        
        # Identificar as classes com desbalanceamento superior a 25%
        unique_classes, counts = np.unique(train_y, return_counts=True)
        max_count = counts.max()
        target_classes = unique_classes[counts / max_count < 0.75]
        
        if len(target_classes) > 0:
            # Aplicar augmentação nas classes com desbalanceamento superior a 25%
            augmentor = fema_augmentor.FEMaAugmentor(
                k=0, basis=fema_classifier.Basis.shepardBasis, th=0.75, target_classes=target_classes
            )
            augmentor.fit(train_x, train_y)
            new_samples, new_labels = augmentor.augment(N=10*int(len(target_classes) * counts.max()))
            train_x = np.vstack([train_x, new_samples])
            train_y = np.vstack([train_y, new_labels])
        
        print(f"AFTER Evaluating dataset")
        print(f"Sample counts per class: {np.bincount(train_y[:,0])}")
        
    train_y = train_y.ravel()  # Ajustar para o formato esperado pelos classificadores

    # Ajustar modelo conforme o classificador escolhido
    if classifier == 'fema':
        
        train_y_fema = train_y.reshape((train_y.shape[0],1))
        model = fema_classifier.FEMaClassifier(k=2, basis=fema_classifier.Basis.shepardBasis)
        model.fit(train_x, train_y_fema)
        pred, _ = model.predict(test_x, 10)
    elif classifier == 'rf':
        model = RandomForestClassifier()
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
    elif classifier == 'svm':
        model = SVC()
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
    elif classifier == 'logistic':
        model = LogisticRegression()
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
    else:
        raise ValueError("Unknown classifier")

    accuracy = accuracy_score(test_y, pred)
    balanced_accuracy = balanced_accuracy_score(test_y, pred)
    precision = precision_score(test_y, pred, average='weighted')
    recall = recall_score(test_y, pred, average='weighted')
    f1 = f1_score(test_y, pred, average='weighted')

    # Métricas de avaliação    
    print(f"{classifier.upper()} {'With' if augment else 'Without'} Augmentation:")
    print(f"Confusion Matrix:\n{confusion_matrix(test_y, pred)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print('-' * 50)
# Função principal para testar o modelo em diferentes datasets da OpenML
def main():
    dataset_ids = [
        #31,    # credit-g
        1464,  # kc1
        37,    # diabetes
        #15,    # kr-vs-kp
        40983, # electricity
        3917,  # wine-quality-red
        #1049,  # ozone-level-8hr
        #38,    # abalone
        #179,  # adult
        #554,  # car
        #1590,  # churn
        #42132, # bank-marketing
        #42193, # phoneme
        #23,    # heart-h
        #300,   # segment
        #44,    # sonar
        #1038,  # cylinder-bands
        #4534,  # numerai28.6
        #4538,  # numerai28.6
        #40685  # madelon
    ]


    classifiers = ['rf', 'svm', 'logistic','fema'] 

    for dataset_id in dataset_ids:
        dataset = fetch_openml(data_id=dataset_id)

        label_encoder = LabelEncoder()
        
        # Tratar features do tipo string
        dataset.data = handle_string_features(pd.DataFrame(dataset.data))

        # Converta os labels de texto para números
        dataset.target = label_encoder.fit_transform(dataset.target)
        

        print(f"Evaluating dataset: {dataset_id}")
        print(f"Sample counts per class: {np.bincount(dataset.target)}")
        
        train_x, test_x, train_y, test_y = train_test_split(dataset.data, dataset.target, test_size=0.25)


        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        
        for clf in classifiers:
            evaluate_model(train_x, train_y, test_x, test_y, classifier=clf, augment=False)
            evaluate_model(train_x, train_y, test_x, test_y, classifier=clf, augment=True)

        
if __name__ == "__main__":
    main()
