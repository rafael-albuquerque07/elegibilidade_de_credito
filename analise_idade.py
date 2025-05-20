#!/usr/bin/env python3
"""
Script para analisar a importância da feature "idade" no modelo de
elegibilidade de crédito.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuração de visualização
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def carregar_dados(caminho_arquivo='data/elegibilidade_credito.csv'):
    """Carrega os dados do arquivo CSV"""
    df = pd.read_csv(caminho_arquivo)
    print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df

def processar_historico_pagamento(df):
    """Converte a coluna de histórico de pagamento para formato numérico"""
    df['historico_pagamento'] = df['historico_pagamento (score)'].copy()
    
    # Converter string para float
    if df['historico_pagamento'].dtype == 'object':
        df['historico_pagamento'] = df['historico_pagamento'].str.replace('.', '')
        df['historico_pagamento'] = df['historico_pagamento'].str.replace(',', '.')
        df['historico_pagamento'] = pd.to_numeric(df['historico_pagamento']) / 10000000000000000
    
    return df

def criar_features_derivadas(df):
    """Cria features derivadas a partir das existentes"""
    # Razão de endividamento
    df['razao_endividamento'] = df['total_dividas'] / df['salario_anual']
    
    # Capacidade de pagamento
    df['capacidade_pagamento'] = (df['salario_anual'] - df['total_dividas']) / df['credito_solicitado']
    
    return df

def analisar_correlacoes(df):
    """Analisa correlações entre features, incluindo idade"""
    # Selecionar features numéricas
    features_numericas = ['salario_anual', 'total_dividas', 'historico_pagamento', 
                        'idade', 'credito_solicitado', 'razao_endividamento', 
                        'capacidade_pagamento', 'elegibilidade']
    
    # Calcular correlações
    corr = df[features_numericas].corr()
    
    # Visualizar matriz de correlação
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    
    plt.title('Matriz de Correlação (incluindo Idade)', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizacoes/matriz_correlacao_com_idade.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Destacar correlação da idade com elegibilidade
    print(f"\nCorrelação entre idade e elegibilidade: {corr.loc['idade', 'elegibilidade']:.4f}")
    return corr

def visualizar_idade_por_categoria(df):
    """Visualiza a distribuição de idade por categoria de elegibilidade"""
    plt.figure(figsize=(12, 6))
    
    # Boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(x='elegibilidade', y='idade', data=df, palette='viridis')
    plt.title('Distribuição de Idade por Categoria', fontsize=14)
    plt.xlabel('Categoria de Elegibilidade', fontsize=12)
    plt.ylabel('Idade', fontsize=12)
    plt.xticks([0, 1, 2], ['Não Elegível', 'Elegível c/ Análise', 'Elegível'])
    
    # Violinplot para melhor visualização da distribuição
    plt.subplot(1, 2, 2)
    sns.violinplot(x='elegibilidade', y='idade', data=df, palette='viridis', inner='quartile')
    plt.title('Distribuição de Idade por Categoria (Violinplot)', fontsize=14)
    plt.xlabel('Categoria de Elegibilidade', fontsize=12)
    plt.ylabel('Idade', fontsize=12)
    plt.xticks([0, 1, 2], ['Não Elegível', 'Elegível c/ Análise', 'Elegível'])
    
    plt.tight_layout()
    plt.savefig('visualizacoes/idade_por_categoria.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Estatísticas por grupo
    stats = df.groupby('elegibilidade')['idade'].agg(['mean', 'median', 'std', 'min', 'max'])
    stats.index = ['Não Elegível', 'Elegível c/ Análise', 'Elegível']
    print("\nEstatísticas de idade por categoria:")
    print(stats)
    
    return stats

def treinar_e_comparar_modelos(df, random_state=42, test_size=0.2, k=21):
    """Treina e compara modelos com e sem a feature idade"""
    # Features base (sem idade)
    features_base = ['salario_anual', 'total_dividas', 'historico_pagamento', 
                  'razao_endividamento', 'capacidade_pagamento']
    
    # Features com idade
    features_com_idade = features_base + ['idade']
    
    # Target
    y = df['elegibilidade']
    
    # Preparar conjuntos de dados
    X_base = df[features_base]
    X_com_idade = df[features_com_idade]
    
    # Dividir em treino e teste
    X_base_train, X_base_test, y_train, y_test = train_test_split(
        X_base, y, test_size=test_size, random_state=random_state
    )
    
    X_idade_train, X_idade_test, _, _ = train_test_split(
        X_com_idade, y, test_size=test_size, random_state=random_state
    )
    
    # Normalizar
    scaler_base = StandardScaler()
    X_base_train_norm = scaler_base.fit_transform(X_base_train)
    X_base_test_norm = scaler_base.transform(X_base_test)
    
    scaler_idade = StandardScaler()
    X_idade_train_norm = scaler_idade.fit_transform(X_idade_train)
    X_idade_test_norm = scaler_idade.transform(X_idade_test)
    
    # Treinar modelos KNN
    modelo_base = KNeighborsClassifier(n_neighbors=k)
    modelo_base.fit(X_base_train_norm, y_train)
    
    modelo_com_idade = KNeighborsClassifier(n_neighbors=k)
    modelo_com_idade.fit(X_idade_train_norm, y_train)
    
    # Avaliar modelos
    y_pred_base = modelo_base.predict(X_base_test_norm)
    accuracy_base = accuracy_score(y_test, y_pred_base)
    
    y_pred_idade = modelo_com_idade.predict(X_idade_test_norm)
    accuracy_idade = accuracy_score(y_test, y_pred_idade)
    
    # Mostrar resultados
    print("\n===== COMPARAÇÃO DE MODELOS =====")
    print(f"Acurácia do modelo SEM idade: {accuracy_base:.4f}")
    print(f"Acurácia do modelo COM idade: {accuracy_idade:.4f}")
    print(f"Diferença: {(accuracy_idade - accuracy_base):.4f}")
    
    # Testar diferentes valores de k para verificar consistência
    k_values = [5, 7, 11, 15, 21, 31, 41, 51]
    accuracies_base = []
    accuracies_idade = []
    
    for k_val in k_values:
        # Modelo base
        model = KNeighborsClassifier(n_neighbors=k_val)
        model.fit(X_base_train_norm, y_train)
        y_pred = model.predict(X_base_test_norm)
        accuracies_base.append(accuracy_score(y_test, y_pred))
        
        # Modelo com idade
        model = KNeighborsClassifier(n_neighbors=k_val)
        model.fit(X_idade_train_norm, y_train)
        y_pred = model.predict(X_idade_test_norm)
        accuracies_idade.append(accuracy_score(y_test, y_pred))
    
    # Visualizar resultados para diferentes k
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, accuracies_base, 'o-', label='Modelo SEM idade')
    plt.plot(k_values, accuracies_idade, 's-', label='Modelo COM idade')
    plt.xlabel('Valor de K', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.title('Comparação de Acurácia para Diferentes Valores de K', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig('visualizacoes/comparacao_acuracia_idade.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Verificar importância relativa das features no modelo com idade
    if hasattr(modelo_com_idade, 'feature_importances_'):
        importances = modelo_com_idade.feature_importances_
        print("\nImportância das features:")
        for feature, importance in zip(features_com_idade, importances):
            print(f"  - {feature}: {importance:.4f}")
    else:
        # KNN não tem atributo feature_importances_, então usamos um método alternativo
        print("\nKNN não fornece feature importances diretamente.")
        
    # Matriz de confusão para modelo com idade
    cm = confusion_matrix(y_test, y_pred_idade)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Não Elegível', 'Elegível c/ Análise', 'Elegível'],
                yticklabels=['Não Elegível', 'Elegível c/ Análise', 'Elegível'])
    plt.title('Matriz de Confusão (Modelo com Idade)', fontsize=14)
    plt.xlabel('Previsto', fontsize=12)
    plt.ylabel('Verdadeiro', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizacoes/matriz_confusao_com_idade.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Relatório de classificação
    print("\nRelatório de classificação (Modelo COM idade):")
    print(classification_report(y_test, y_pred_idade, 
                              target_names=['Não Elegível', 'Elegível c/ Análise', 'Elegível']))
    
    return {
        'accuracy_base': accuracy_base,
        'accuracy_idade': accuracy_idade,
        'diff': accuracy_idade - accuracy_base,
        'k_values': k_values,
        'accuracies_base': accuracies_base,
        'accuracies_idade': accuracies_idade
    }

def criar_segmentacao_idade(df):
    """Cria segmentação de idade em faixas para análise"""
    bins = [18, 30, 40, 50, 60, 100]
    labels = ['18-30', '31-40', '41-50', '51-60', '61+']
    df['faixa_idade'] = pd.cut(df['idade'], bins=bins, labels=labels, right=False)
    
    # Visualizar distribuição de elegibilidade por faixa de idade
    plt.figure(figsize=(14, 6))
    
    # Contagem absoluta
    plt.subplot(1, 2, 1)
    sns.countplot(x='faixa_idade', hue='elegibilidade', data=df, palette='viridis',
                 hue_order=[1, 2, 3])
    plt.title('Distribuição de Elegibilidade por Faixa Etária', fontsize=14)
    plt.xlabel('Faixa Etária', fontsize=12)
    plt.ylabel('Contagem', fontsize=12)
    plt.legend(title='Elegibilidade', labels=['Não Elegível', 'Elegível c/ Análise', 'Elegível'])
    
    # Proporção normalizada
    plt.subplot(1, 2, 2)
    prop_data = df.groupby(['faixa_idade', 'elegibilidade']).size().unstack()
    prop_data = prop_data.div(prop_data.sum(axis=1), axis=0) * 100
    prop_data.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Proporção de Elegibilidade por Faixa Etária (%)', fontsize=14)
    plt.xlabel('Faixa Etária', fontsize=12)
    plt.ylabel('Porcentagem', fontsize=12)
    plt.legend(title='Elegibilidade', labels=['Não Elegível', 'Elegível c/ Análise', 'Elegível'])
    
    plt.tight_layout()
    plt.savefig('visualizacoes/elegibilidade_por_faixa_etaria.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analisar tabela cruzada
    tabela = pd.crosstab(df['faixa_idade'], df['elegibilidade'], normalize='index') * 100
    tabela.columns = ['Não Elegível', 'Elegível c/ Análise', 'Elegível']
    print("\nDistribuição percentual de elegibilidade por faixa etária:")
    print(tabela)
    
    return tabela

def principal():
    """Função principal que executa a análise completa"""
    # Criar diretório para visualizações
    os.makedirs('visualizacoes', exist_ok=True)
    
    # Carregar e processar dados
    df = carregar_dados()
    df = processar_historico_pagamento(df)
    df = criar_features_derivadas(df)
    
    # Exibir estatísticas básicas da idade
    print("\nEstatísticas da feature idade:")
    print(df['idade'].describe())
    
    # Visualizar distribuição da idade
    plt.figure(figsize=(12, 6))
    sns.histplot(df['idade'], kde=True, bins=30)
    plt.title('Distribuição da Idade dos Solicitantes', fontsize=14)
    plt.xlabel('Idade', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizacoes/distribuicao_idade.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analisar correlações
    corr = analisar_correlacoes(df)
    
    # Visualizar idade por categoria
    stats = visualizar_idade_por_categoria(df)
    
    # Criar segmentação de idade
    tabela = criar_segmentacao_idade(df)
    
    # Treinar e comparar modelos
    resultados = treinar_e_comparar_modelos(df)
    
    # Resumo final
    print("\n========= RESUMO DA ANÁLISE DA FEATURE IDADE =========")
    print(f"- Correlação com elegibilidade: {corr.loc['idade', 'elegibilidade']:.4f}")
    print(f"- Média de idade por categoria:")
    for i, cat in enumerate(['Não Elegível', 'Elegível c/ Análise', 'Elegível']):
        print(f"  * {cat}: {stats.loc[cat, 'mean']:.2f} anos")
    
    print(f"- Desempenho do modelo:")
    print(f"  * Acurácia SEM idade: {resultados['accuracy_base']:.4f}")
    print(f"  * Acurácia COM idade: {resultados['accuracy_idade']:.4f}")
    print(f"  * Diferença: {resultados['diff']:.4f}")
    
    # Recomendação final
    if resultados['diff'] > 0.01:  # Diferença de mais de 1 ponto percentual
        print("\nRECOMENDAÇÃO: Incluir a feature 'idade' no modelo, pois ela melhora")
        print("significativamente o desempenho.")
    elif resultados['diff'] > 0:  # Qualquer melhoria positiva
        print("\nRECOMENDAÇÃO: Considerar incluir a feature 'idade' no modelo, pois ela")
        print("apresenta uma pequena melhoria no desempenho.")
    else:  # Sem melhoria ou piora
        print("\nRECOMENDAÇÃO: Manter o modelo sem a feature 'idade', pois ela não")
        print("contribui para melhorar o desempenho.")

if __name__ == "__main__":
    principal()