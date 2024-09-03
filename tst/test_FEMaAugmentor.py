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
from imblearn.under_sampling import EditedNearestNeighbours as ENN
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from imblearn.metrics import geometric_mean_score

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
def apply_fema_augmentation(train_x, train_y, th, scale, loc, basis, use_smart_sampling=False, apply_enn=False):
    original_counts = np.bincount(train_y)
    train_y = train_y.reshape((train_y.shape[0], 1))
    unique_classes, counts = np.unique(train_y, return_counts=True)
    max_count = counts.max()
    target_classes = unique_classes[counts / max_count < 0.9]
    
    if len(target_classes) > 0:
        augmentor = fema_augmentor.FEMaAugmentor(
            k=0, basis=basis, th_min=th, th_max=th+0.5, scale=scale, loc=loc, 
            target_classes=target_classes, use_smart_sampling=use_smart_sampling, apply_enn=False, z=3
        )
        augmentor.fit(train_x, train_y)
        new_samples, new_labels = augmentor.augment(N=int(len(target_classes) * counts.max()))
        if new_samples.shape[0] > 0: 
            train_x = np.vstack([train_x, new_samples])
            train_y = np.vstack([train_y, new_labels])

    train_y = train_y.ravel()
    augmented_counts = np.bincount(train_y)

    # Aplicar ENN se a flag estiver ativada e houver mais de uma classe no conjunto de dados
    if apply_enn:
        unique_classes_after, _ = np.unique(train_y, return_counts=True)
        if len(unique_classes_after) > 1:
            try:
                enn = ENN()
                train_x, train_y = enn.fit_resample(train_x, train_y)
                augmented_counts = np.bincount(train_y)
            except Exception as e:
                print(f"Error applying ENN: {str(e)}")
        else:
            print("Skipping ENN as only one class is present after augmentation.")

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
    mcc = matthews_corrcoef(test_y, pred)
    gmean = geometric_mean_score(test_y, pred, average='weighted')


    metrics = {
        'classifier': classifier,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_score': f1,
        'per_class_accuracy': per_class_accuracy,
        'mcc' : mcc,
        'gmean': gmean,
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
        #1049,  # ozone-level-8hr
        #23,    # heart-h
        #44,    # sonar
        #4534,  # numerai28.6
        #4538,  # numerai28.6
        #40685  # madelon
    ]
    
    test_sizes = [0.5, 0.25, 0.05]  # Tamanhos percentuais do conjunto de testes
    classifiers = ['rf', 'svm', 'logistic', 'fema']
    fema_parameters = [
        {'th': 0.75, 'scale': 0.50, 'loc': 0.0},
        {'th': 0.90, 'scale': 0.25, 'loc': 0.0},
        {'th': 0.75, 'scale': 0.25, 'loc': 0.0},
        {'th': 0.90, 'scale': 0.12, 'loc': 0.0},
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

    N = 2  # Número de repetições do experimento
    all_results = []

    for iteration in range(1, N + 1):
        for test_size in test_sizes:  # Iterar sobre cada tamanho de conjunto de testes
            for dataset_id in dataset_ids:
                dataset = fetch_openml(data_id=dataset_id)
                label_encoder = LabelEncoder()

                dataset.data = handle_string_features(pd.DataFrame(dataset.data))
                dataset.target = label_encoder.fit_transform(dataset.target)

                print(f"Evaluating dataset: {dataset_id} with test size {test_size}")
                print(f"Sample counts per class: {np.bincount(dataset.target)}")

                train_x, test_x, train_y, test_y = train_test_split(dataset.data, dataset.target, test_size=test_size, stratify=dataset.target)

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
                    metrics['th'] = None
                    metrics['scale'] = None
                    metrics['loc'] = None
                    metrics['enn'] = None
                    metrics['smart'] = None
                    metrics['test_size'] = test_size  # Adicionar o tamanho do conjunto de testes
                    all_results.append(metrics)

                # Avaliar FEMaAugmentor com diferentes parâmetros e bases, incluindo apply_enn
                for params in fema_parameters:
                    for basis in bases:
                        for use_smart in [False, True]:
                            for apply_enn in [False, True]:
                                train_x_augmented, train_y_augmented, original_counts, augmented_counts = apply_fema_augmentation(
                                    train_x, train_y, th=params['th'], scale=params['scale'], loc=params['loc'],
                                    basis=basis, use_smart_sampling=use_smart, apply_enn=apply_enn
                                )
                                for clf in classifiers:
                                    metrics = evaluate_model(train_x_augmented, train_y_augmented, test_x, test_y, classifier=clf)
                                    metrics['dataset_id'] = dataset_id
                                    metrics['iteration'] = iteration
                                    metrics['augmentation'] = f"FEMaAugmentor_{'shepard' if basis == fema_classifier.Basis.shepardBasis else 'radial'}_{'smart' if use_smart else 'default'}_{'enn' if apply_enn else 'no_enn'}"
                                    metrics['original_counts'] = original_counts
                                    metrics['augmented_counts'] = augmented_counts
                                    metrics['th'] = params['th']
                                    metrics['scale'] = params['scale']
                                    metrics['loc'] = params['loc']
                                    metrics['enn'] = 'yes' if apply_enn else 'no'
                                    metrics['smart'] = 'yes' if use_smart else 'no'
                                    metrics['test_size'] = test_size  # Adicionar o tamanho do conjunto de testes
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
                        metrics['th'] = None
                        metrics['scale'] = None
                        metrics['loc'] = None
                        metrics['enn'] = None
                        metrics['smart'] = None
                        metrics['test_size'] = test_size  # Adicionar o tamanho do conjunto de testes
                        all_results.append(metrics)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("evaluation_results.csv", index=False)

    # Carregar os resultados
    results_df = pd.read_csv("evaluation_results.csv")

    # Configurações para os gráficos
    sns.set(style="whitegrid")

    # Obter todos os IDs de datasets únicos
    dataset_ids = results_df['dataset_id'].unique()

    # Obter todas as métricas que estão presentes no DataFrame
    metric_columns = ['balanced_accuracy', 'f1_score']  # Adicione as métricas que você deseja plotar

    # Iterar sobre cada dataset e cada métrica para gerar gráficos específicos
    for dataset_id in dataset_ids:
        for test_size in test_sizes:  # Gerar gráficos para cada tamanho de conjunto de testes
            dataset_results = results_df[(results_df['dataset_id'] == dataset_id) & (results_df['test_size'] == test_size)]
            
            for metric in metric_columns:
                # Criar uma tabela pivô onde linhas são abordagens de desbalanceamento e colunas são classificadores
                pivot_table_mean = dataset_results.pivot_table(
                    index='augmentation', columns='classifier', values=metric, aggfunc='mean'
                )
                
                pivot_table_std = dataset_results.pivot_table(
                    index='augmentation', columns='classifier', values=metric, aggfunc='std'
                )

                # Gerar o gráfico de barras com erro padrão
                plt.figure(figsize=(12, 8))
                pivot_table_mean.plot(kind='bar', yerr=pivot_table_std, capsize=4, ax=plt.gca())
                
                plt.title(f'Comparação de Abordagens de Desbalanceamento por Classificador - Dataset {dataset_id} - Test Size {test_size} - {metric.capitalize()}')
                plt.xlabel('Abordagem de Desbalanceamento')
                plt.ylabel(f'{metric.capitalize()}')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Classificador')
                plt.tight_layout()
                plt.savefig(f'resampling_strategy_vs_classifier_{dataset_id}_{test_size}_{metric}_barplot.png')
                plt.show()

                # Gerar o heatmap para a mesma métrica e dataset
                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_table_mean, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': metric.capitalize()})
                
                plt.title(f'Heatmap de Abordagens de Desbalanceamento por Classificador - Dataset {dataset_id} - Test Size {test_size} - {metric.capitalize()}')
                plt.xlabel('Classificador')
                plt.ylabel('Abordagem de Desbalanceamento')
                plt.tight_layout()
                plt.savefig(f'resampling_strategy_vs_classifier_{dataset_id}_{test_size}_{metric}_heatmap.png')
                plt.show()

if __name__ == "__main__":
    main()