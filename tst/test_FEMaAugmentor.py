import sys
import os
from typing import Tuple

sys.path.append('/home/danillorp/Área de Trabalho/github/fema/src/')
import sys
import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
import fema_classifier
import fema_augmentor

# Função para tratar features do tipo string
def handle_string_features(data):
    string_columns = data.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        print(f"Handling string features: {list(string_columns)}")
        for col in string_columns:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_features = encoder.fit_transform(data[[col]])
            encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out([col]))
            data = pd.concat([data.drop([col], axis=1), encoded_df], axis=1)
    return data

# Função para aplicar FEMaAugmentor com diferentes parâmetros e salvar as contagens
def apply_fema_augmentation(train_x, train_y, th, scale, loc, basis, use_smart_sampling=False):
    original_counts = np.bincount(train_y)
    train_y = train_y.reshape((train_y.shape[0], 1))
    unique_classes, counts = np.unique(train_y, return_counts=True)
    max_count = counts.max()
    target_classes = unique_classes[counts / max_count < 0.9]
    
    if len(target_classes) > 0:
        augmentor = fema_augmentor.FEMaAugmentor(
            k=0, basis=basis, th=th, scale=scale, loc=loc, 
            target_classes=target_classes, use_smart_sampling=use_smart_sampling, z=2
        )
        augmentor.fit(train_x, train_y)
        new_samples, new_labels = augmentor.augment(N=int(len(target_classes) * counts.max()))
        if new_samples.shape[0] > 0: 
            train_x = np.vstack([train_x, new_samples])
            train_y = np.vstack([train_y, new_labels])
    
    train_y = train_y.ravel()
    augmented_counts = np.bincount(train_y)
    return train_x, train_y, original_counts, augmented_counts

# Função para aplicar outras estratégias de tratamento de desbalanceamento e salvar as contagens
def apply_resampling_strategy(strategy, train_x, train_y):
    original_counts = np.bincount(train_y)
    try:
        train_x_resampled, train_y_resampled = strategy.fit_resample(train_x, train_y)
        resampled_counts = np.bincount(train_y_resampled)
    except ValueError:
        train_x_resampled, train_y_resampled = train_x, train_y
        resampled_counts = original_counts
    
    return train_x_resampled, train_y_resampled, original_counts, resampled_counts

# Função para avaliar o modelo e salvar métricas
def evaluate_model(train_x, train_y, test_x, test_y, classifier):
    if classifier == 'fema':
        train_y_fema = train_y.reshape((train_y.shape[0], 1))
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
    per_class_accuracy = accuracy_score(test_y, pred, normalize=False) / len(np.unique(test_y))

    metrics = {
        'classifier': classifier,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_score': f1,
        'per_class_accuracy': per_class_accuracy,
    }

    print(f"{classifier.upper()} Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print('-' * 50)

    return metrics

# Função principal para testar o modelo em diferentes datasets da OpenML
def main():
    dataset_ids = [   
        1464,  # kc1
        37,    # diabetes
        1049,  # ozone-level-8hr
        23,    # heart-h
        44,    # sonar
        #4534,  # numerai28.6
        #4538,  # numerai28.6
        #40685  # madelon
    ]
    
    
    classifiers = ['rf', 'svm', 'logistic', 'fema'] 
    fema_parameters = [
        {'th': 0.5,     'scale': 1.00,   'loc': 0.0},
        {'th': 0.75,    'scale': 0.50,   'loc': 0.0},
        {'th': 0.95,    'scale': 0.25,   'loc': 0.0},
        {'th': 0.5,     'scale': 0.50,   'loc': 0.0},
        {'th': 0.75,    'scale': 0.25,   'loc': 0.0},
        {'th': 0.95,    'scale': 0.12,   'loc': 0.0},
    ]
    
    bases = [
        fema_classifier.Basis.shepardBasis,
        fema_classifier.Basis.radialBasis,
    ]
    
    resampling_strategies = {
        'No Resampling': None,
        'SMOTE': SMOTE(),
        'ADASYN': ADASYN(),
        'Random OverSampling': RandomOverSampler(),
        'Random UnderSampling': RandomUnderSampler(),
        'Tomek Links': TomekLinks(),
        'NearMiss': NearMiss(),
        'SMOTEENN': SMOTEENN(),
        'SMOTETomek': SMOTETomek(),
    }

    N = 1  # Número de repetições do experimento
    all_results = []

    for iteration in range(1, N + 1):
        for dataset_id in dataset_ids:
            dataset = fetch_openml(data_id=dataset_id)
            label_encoder = LabelEncoder()
            
            dataset.data = handle_string_features(pd.DataFrame(dataset.data))
            dataset.target = label_encoder.fit_transform(dataset.target)
            
            print(f"Evaluating dataset: {dataset_id}")
            print(f"Sample counts per class: {np.bincount(dataset.target)}")
            
            train_x, test_x, train_y, test_y = train_test_split(dataset.data, dataset.target, test_size=0.25)

            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)

            # Avaliar sem resampling
            for clf in classifiers:
                metrics = evaluate_model(train_x, train_y, test_x, test_y, classifier=clf)
                metrics['dataset_id'] = dataset_id
                metrics['iteration'] = iteration
                metrics['augmentation'] = 'None'
                metrics['original_counts'] = np.bincount(train_y)
                all_results.append(metrics)

            # Avaliar FEMaAugmentor com diferentes parâmetros e bases
            for params in fema_parameters:
                for basis in bases:
                    for use_smart in [False, True]:
                        train_x_augmented, train_y_augmented, original_counts, augmented_counts = apply_fema_augmentation(
                            train_x, train_y, th=params['th'], scale=params['scale'], loc=params['loc'], basis=basis, use_smart_sampling=use_smart
                        )
                        for clf in classifiers:
                            metrics = evaluate_model(train_x_augmented, train_y_augmented, test_x, test_y, classifier=clf)
                            metrics['dataset_id'] = dataset_id
                            metrics['iteration'] = iteration
                            smart_label = 'smart' if use_smart else 'default'
                            basis_label = 'shepard' if basis == fema_classifier.Basis.shepardBasis else 'radial'
                            metrics['augmentation'] = f"FEMaAugmentor_{basis_label}_{smart_label}_th{params['th']}_scale{params['scale']}_loc{params['loc']}"
                            metrics['original_counts'] = original_counts
                            metrics['augmented_counts'] = augmented_counts
                            all_results.append(metrics)

            # Avaliar as demais estratégias de resampling
            for strategy_name, strategy in resampling_strategies.items():
                if strategy is None:
                    continue  # Já avaliamos o caso de 'No Resampling' anteriormente
                
                train_x_resampled, train_y_resampled, original_counts, resampled_counts = apply_resampling_strategy(
                    strategy, train_x, train_y
                )
                for clf in classifiers:
                    metrics = evaluate_model(train_x_resampled, train_y_resampled, test_x, test_y, classifier=clf)
                    metrics['dataset_id'] = dataset_id
                    metrics['iteration'] = iteration
                    metrics['augmentation'] = strategy_name
                    metrics['original_counts'] = original_counts
                    metrics['augmented_counts'] = resampled_counts
                    all_results.append(metrics)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("evaluation_results.csv", index=False)

if __name__ == "__main__":
    main()
