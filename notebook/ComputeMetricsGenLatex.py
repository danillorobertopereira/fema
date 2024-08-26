import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Carregar os dados do CSV
#file_path = "/home/danillorp/Área de Trabalho/github/fema/notebook/clustering_results-FINAL-TOY.xlsx"
#file_path = "/home/danillorp/Área de Trabalho/github/fema/notebook/clustering_results-real-final.xlsx"

file_path = "/home/danillorp/Área de Trabalho/github/fema/notebook/expanded_dataset.xlsx"

data = pd.read_excel(file_path)

data = data.dropna()

# Calcula a média e o desvio padrão para cada dataset e método
summary = data.groupby(['Dataset', 'Method']).agg({
    'Time': ['mean', 'std'],
    'Silhouette Score': ['mean', 'std'],
    'Davies-Bouldin Score': ['mean', 'std'],
    'Adjusted Rand Index': ['mean', 'std'],
    'Normalized Mutual Information': ['mean', 'std']
}).reset_index()

summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
summary.rename(columns={'Dataset_': 'Dataset', 'Method_': 'Method'}, inplace=True)

# Realiza testes t de duas amostras para cada métrica
metrics = ['Silhouette Score', 'Davies-Bouldin Score', 'Adjusted Rand Index', 'Normalized Mutual Information']
results = []

for dataset in data['Dataset'].unique():
    for metric in metrics:
        methods = data[data['Dataset'] == dataset]['Method'].unique()
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1 = methods[i]
                method2 = methods[j]
                group1 = data[(data['Dataset'] == dataset) & (data['Method'] == method1)][metric]
                group2 = data[(data['Dataset'] == dataset) & (data['Method'] == method2)][metric]
                t_stat, p_val = ttest_ind(group1.dropna(), group2.dropna(), equal_var=False)
                results.append({
                    'Dataset': dataset,
                    'Metric': metric,
                    'Method1': method1,
                    'Method2': method2,
                    't_stat': t_stat,
                    'p_val': p_val
                })

results_df = pd.DataFrame(results)

# Determina os melhores métodos com base na média das métricas
best_methods = summary.loc[summary.groupby('Dataset')[['Silhouette Score_mean', 'Davies-Bouldin Score_mean', 'Adjusted Rand Index_mean', 'Normalized Mutual Information_mean']].idxmax().values.flatten()]

# Gera tabelas separadas em LaTeX para cada métrica
def to_latex_highlight(df, best_methods, metric):
    df[f'{metric}_combined'] = df.apply(lambda x: f"{x[f'{metric}_mean']:.2f}±{x[f'{metric}_std']:.2f}", axis=1)
    df_metric = df.pivot(index='Method', columns='Dataset', values=f'{metric}_combined')
    
    # Adiciona a legenda
    caption = f"Performance metrics for {metric.replace('_', ' ')}"
    latex_str = "\\begin{table}[ht]\n\\centering\n\\begin{adjustbox}{max width=\\textwidth}\n" \
                + df_metric.to_latex(escape=False) \
                + "\\end{adjustbox}\n\\caption{" + caption + "}\n\\label{tab:" + metric.replace(" ", "_") + "}\n\\end{table}"
    
    return latex_str

for metric in metrics:
    latex_table = to_latex_highlight(summary, best_methods, metric)
    with open(f'/home/danillorp/Área de Trabalho/github/fema/notebook/summary_real_table_{metric.replace(" ", "_")}.tex', 'w') as f:
        f.write(latex_table)
    print(f"Tabela LaTeX para {metric} gerada e salva em '/home/danillorp/Área de Trabalho/github/fema/notebook/summary_table_{metric.replace(' ', '_')}.tex'")

# Exibe o caminho dos arquivos gerados
print("Tabelas LaTeX geradas com sucesso.")
