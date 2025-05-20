#!/usr/bin/env python3
"""
Script simplificado para analisar a importância da feature "idade" no modelo de
elegibilidade de crédito - sem visualizações complexas.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def carregar_dados(caminho_arquivo='data/elegibilidade_credito.csv'):
    """Carrega os dados do arquivo CSV"""
    print("Carregando dados...")
    df = pd.read_csv(caminho_arquivo)
    print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df

def processar_dados(df):
    """Processa os dados básicos"""
    print("Processando dados...")
    
    # Processar histórico de pagamento
    df['historico_pagamento'] = df['historico_pagamento (score)'].copy()
    if df['historico_pagamento'].dtype == 'object':
        df['historico_pagamento'] = df['historico_pagamento'].str.replace('.', '')
        df['historico_pagamento'] = df['historico_pagamento'].str.replace(',', '.')
        df['historico_pagamento'] = pd.to_numeric(df['historico_pagamento']) / 10000000000000000
    
    # Criar features derivadas
    df['razao_endividamento'] = df['total_dividas'] / df['salario_anual']
    df['capacidade_pagamento'] = (df['salario_anual'] - df['total_dividas']) / df['credito_solicitado']
    
    print("Dados processados com sucesso.")
    return df

def analisar_correlacao_idade(df):
    """Analisa a correlação da idade com elegibilidade"""
    print("\nAnalisando correlação da idade...")
    
    # Calcular correlação da idade com elegibilidade
    corr_idade_elegib = df['idade'].corr(df['elegibilidade'])
    print(f"Correlação entre idade e elegibilidade: {corr_idade_elegib:.4f}")
    
    # Features numéricas para correlação
    features_num = ['salario_anual', 'total_dividas', 'historico_pagamento', 
                   'idade', 'credito_solicitado', 'razao_endividamento', 
                   'capacidade_pagamento', 'elegibilidade']
    
    # Matriz de correlação
    corr_matrix = df[features_num].corr()
    
    # Ranking de correlação com elegibilidade (valor absoluto)
    correlacoes = corr_matrix['elegibilidade'].abs().sort_values(ascending=False)
    print("\nRanking de correlação com elegibilidade (valor absoluto):")
    for feature, value in correlacoes.items():
        if feature != 'elegibilidade':  # Excluir autocorrelação
            print(f"  - {feature}: {corr_matrix['elegibilidade'][feature]:.4f}")
    
    return corr_idade_elegib, correlacoes

def analisar_estatisticas_por_categoria(df):
    """Analisa estatísticas de idade por categoria de elegibilidade"""
    print("\nAnalisando idade por categoria de elegibilidade...")
    
    # Estatísticas por grupo
    stats = df.groupby('elegibilidade')['idade'].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
    
    # Renomear índices para clareza
    stats.index = ['Não Elegível', 'Elegível c/ Análise', 'Elegível']
    
    print("\nEstatísticas de idade por categoria:")
    print(stats)
    
    # Calcular diferença percentual entre médias
    min_mean = stats['mean'].min()
    max_mean = stats['mean'].max()
    diff_pct = (max_mean - min_mean) / min_mean * 100
    
    print(f"\nDiferença percentual entre médias: {diff_pct:.2f}%")
    
    return stats

def comparar_modelos_simples(df):
    """Treina e compara modelos com e sem a feature idade - versão simples"""
    print("\nComparando modelos com e sem a feature idade...")
    
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
    
    # Resultados para diferentes seeds
    seeds = [42, 123, 456, 789, 101112]
    results = []
    
    for seed in seeds:
        # Dividir em treino e teste
        X_base_train, X_base_test, y_train, y_test = train_test_split(
            X_base, y, test_size=0.2, random_state=seed
        )
        
        X_idade_train, X_idade_test, _, _ = train_test_split(
            X_com_idade, y, test_size=0.2, random_state=seed
        )
        
        # Normalizar
        scaler_base = StandardScaler()
        X_base_train_norm = scaler_base.fit_transform(X_base_train)
        X_base_test_norm = scaler_base.transform(X_base_test)
        
        scaler_idade = StandardScaler()
        X_idade_train_norm = scaler_idade.fit_transform(X_idade_train)
        X_idade_test_norm = scaler_idade.transform(X_idade_test)
        
        # Lista para armazenar resultados com diferentes k
        k_results = []
        
        # Testando para diferentes valores de k
        for k in [5, 21, 41]:
            # Modelo base (sem idade)
            modelo_base = KNeighborsClassifier(n_neighbors=k)
            modelo_base.fit(X_base_train_norm, y_train)
            y_pred_base = modelo_base.predict(X_base_test_norm)
            acc_base = accuracy_score(y_test, y_pred_base)
            
            # Modelo com idade
            modelo_idade = KNeighborsClassifier(n_neighbors=k)
            modelo_idade.fit(X_idade_train_norm, y_train)
            y_pred_idade = modelo_idade.predict(X_idade_test_norm)
            acc_idade = accuracy_score(y_test, y_pred_idade)
            
            k_results.append({
                'k': k,
                'acc_base': acc_base,
                'acc_idade': acc_idade,
                'diff': acc_idade - acc_base,
                'diff_pct': (acc_idade - acc_base) / acc_base * 100
            })
        
        # Adicionar média dos resultados para a seed atual
        avg_acc_base = sum(r['acc_base'] for r in k_results) / len(k_results)
        avg_acc_idade = sum(r['acc_idade'] for r in k_results) / len(k_results)
        avg_diff = sum(r['diff'] for r in k_results) / len(k_results)
        avg_diff_pct = sum(r['diff_pct'] for r in k_results) / len(k_results)
        
        results.append({
            'seed': seed,
            'avg_acc_base': avg_acc_base,
            'avg_acc_idade': avg_acc_idade,
            'avg_diff': avg_diff,
            'avg_diff_pct': avg_diff_pct,
            'k_results': k_results
        })
    
    # Média geral de todos os experimentos
    avg_overall_base = sum(r['avg_acc_base'] for r in results) / len(results)
    avg_overall_idade = sum(r['avg_acc_idade'] for r in results) / len(results)
    avg_overall_diff = sum(r['avg_diff'] for r in results) / len(results)
    avg_overall_diff_pct = sum(r['avg_diff_pct'] for r in results) / len(results)
    
    # Exibir resultados
    print("\n===== COMPARAÇÃO DE MODELOS =====")
    print(f"Média geral - acurácia SEM idade: {avg_overall_base:.4f}")
    print(f"Média geral - acurácia COM idade: {avg_overall_idade:.4f}")
    print(f"Diferença média: {avg_overall_diff:.4f} ({avg_overall_diff_pct:.2f}%)")
    print("\nResultados por seed:")
    
    for r in results:
        print(f"\nSeed {r['seed']}:")
        print(f"  - Média acurácia SEM idade: {r['avg_acc_base']:.4f}")
        print(f"  - Média acurácia COM idade: {r['avg_acc_idade']:.4f}")
        print(f"  - Diferença média: {r['avg_diff']:.4f} ({r['avg_diff_pct']:.2f}%)")
        print("  - Resultados por k:")
        
        for kr in r['k_results']:
            print(f"    k={kr['k']}: SEM idade={kr['acc_base']:.4f}, COM idade={kr['acc_idade']:.4f}, diff={kr['diff']:.4f} ({kr['diff_pct']:.2f}%)")
    
    # Contagem de quantas vezes o modelo com idade superou o modelo base
    wins_idade = sum(1 for r in results for kr in r['k_results'] if kr['diff'] > 0)
    total_comparacoes = len(results) * len(results[0]['k_results'])
    win_rate = wins_idade / total_comparacoes * 100
    
    print(f"\nModelo COM idade superou o modelo SEM idade em {wins_idade} de {total_comparacoes} comparações ({win_rate:.2f}%)")
    
    return {
        'avg_acc_base': avg_overall_base,
        'avg_acc_idade': avg_overall_idade,
        'avg_diff': avg_overall_diff,
        'avg_diff_pct': avg_overall_diff_pct,
        'wins_idade': wins_idade,
        'total_comparacoes': total_comparacoes,
        'win_rate': win_rate,
        'results': results
    }

def principal():
    """Função principal que executa a análise simplificada"""
    print("===== ANÁLISE SIMPLIFICADA DA FEATURE IDADE =====")
    
    # Carregar e processar dados
    df = carregar_dados()
    df = processar_dados(df)
    
    # Exibir estatísticas básicas da idade
    print("\nEstatísticas da feature idade:")
    print(df['idade'].describe())
    
    # Analisar correlação
    corr_idade, correlacoes = analisar_correlacao_idade(df)
    
    # Analisar estatísticas por categoria
    stats = analisar_estatisticas_por_categoria(df)
    
    # Comparar modelos
    resultados = comparar_modelos_simples(df)
    
    # Resumo final
    print("\n========= RESUMO DA ANÁLISE DA FEATURE IDADE =========")
    print(f"- Correlação com elegibilidade: {corr_idade:.4f}")
    print(f"- Posição no ranking de correlações: {list(correlacoes.index).index('idade') + 1} de {len(correlacoes) - 1}")
    print(f"- Média de idade por categoria:")
    for cat in stats.index:
        print(f"  * {cat}: {stats.loc[cat, 'mean']:.2f} anos (n={stats.loc[cat, 'count']})")
    
    print(f"- Desempenho médio dos modelos:")
    print(f"  * Acurácia SEM idade: {resultados['avg_acc_base']:.4f}")
    print(f"  * Acurácia COM idade: {resultados['avg_acc_idade']:.4f}")
    print(f"  * Diferença: {resultados['avg_diff']:.4f} ({resultados['avg_diff_pct']:.2f}%)")
    print(f"  * Taxa de vitória do modelo COM idade: {resultados['win_rate']:.2f}%")
    
    # Recomendação final
    threshold_pct = 0.5  # Limiar de 0.5% de melhoria para recomendar a inclusão
    
    print("\n===== RECOMENDAÇÃO FINAL =====")
    if resultados['avg_diff_pct'] > threshold_pct and resultados['win_rate'] > 60:
        print("INCLUIR a feature 'idade' no modelo.")
        print(f"Justificativa: Melhora consistente de {resultados['avg_diff_pct']:.2f}% na acurácia,")
        print(f"com o modelo COM idade superando o modelo SEM idade em {resultados['win_rate']:.2f}% dos casos.")
        
        print("\nPara implementar esta mudança:")
        print("1. Em analise_elegibilidade_credito.ipynb (célula 8af5452e):")
        print("   Adicionar 'idade' a SELECTED_FEATURES")
        print("2. Em main.py (linha ~83):")
        print("   Adicionar 'idade' a features_selecionadas")
    elif resultados['avg_diff_pct'] > 0 and resultados['win_rate'] > 50:
        print("CONSIDERAR incluir a feature 'idade' no modelo.")
        print(f"Justificativa: Pequena melhora de {resultados['avg_diff_pct']:.2f}% na acurácia,")
        print(f"com o modelo COM idade superando o modelo SEM idade em {resultados['win_rate']:.2f}% dos casos.")
        print("Recomenda-se fazer testes adicionais em diferentes amostras de dados.")
    else:
        print("NÃO incluir a feature 'idade' no modelo.")
        print(f"Justificativa: Melhoria de apenas {resultados['avg_diff_pct']:.2f}% na acurácia,")
        print(f"com o modelo COM idade superando o modelo SEM idade em apenas {resultados['win_rate']:.2f}% dos casos.")
        print("O custo computacional adicional não justifica a pequena melhoria.")

if __name__ == "__main__":
    try:
        principal()
    except Exception as e:
        print(f"\nERRO DURANTE A EXECUÇÃO: {str(e)}")
        print("Por favor, verifique os caminhos dos arquivos e as dependências.")