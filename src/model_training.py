import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from joblib import dump
import logging
import os
import datetime
from pathlib import Path

# Configurar logger
logger = logging.getLogger(__name__)

def treinar_knn(X_train, y_train, k=21):
    """
    Treina um modelo KNN com o valor de k fornecido
    """
    try:
        logger.info(f"Treinando modelo KNN com k={k}")
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        logger.info(f"Modelo KNN treinado com sucesso")
        return model
    except Exception as e:
        logger.error(f"Erro ao treinar modelo KNN: {str(e)}")
        raise

def avaliar_knn_cross_validation(X_train, y_train, k_values=None, cv=5):
    """
    Avalia diferentes valores de k para KNN usando validação cruzada
    """
    if k_values is None:
        k_values = [5, 7, 11, 15, 21, 31, 41, 51]
    
    logger.info(f"Avaliando KNN com validação cruzada para k={k_values}")
    resultados = {}
    
    try:
        for k in k_values:
            model = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            mean_score = scores.mean()
            resultados[k] = mean_score
            logger.info(f"KNN k={k}: acurácia média CV={mean_score:.4f}, std={scores.std():.4f}")
        
        # Encontrar o melhor k
        melhor_k = max(resultados, key=resultados.get)
        logger.info(f"Melhor valor de k: {melhor_k} com acurácia: {resultados[melhor_k]:.4f}")
        
        return resultados, melhor_k
    except Exception as e:
        logger.error(f"Erro na validação cruzada do KNN: {str(e)}")
        raise

def treinar_kmeans(X_train, n_clusters=3, random_state=42):
    """
    Treina um modelo K-Means com o número de clusters especificado
    """
    try:
        logger.info(f"Treinando modelo K-Means com {n_clusters} clusters")
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        model.fit(X_train)
        logger.info(f"Modelo K-Means treinado com sucesso")
        return model
    except Exception as e:
        logger.error(f"Erro ao treinar modelo K-Means: {str(e)}")
        raise

def descrever_clusters_kmeans(model, X_train, y_train, df, features_selecionadas, mapeamento_clusters):
    """
    Gera uma descrição detalhada de cada cluster com estatísticas e exemplos representativos
    
    Parâmetros:
    -----------
    model : KMeans
        Modelo K-Means treinado
    X_train : numpy.ndarray
        Dados de treinamento normalizados
    y_train : numpy.ndarray ou pandas.Series
        Rótulos verdadeiros
    df : pandas.DataFrame
        DataFrame original com todos os dados
    features_selecionadas : list
        Lista de nomes das features
    mapeamento_clusters : dict
        Mapeamento de cluster para classe
        
    Retorna:
    --------
    dict
        Dicionário com descrições dos clusters
    """
    try:   
        logger.info("Gerando descrição detalhada dos clusters")
        
        # Obter labels dos clusters para dados de treinamento
        clusters = model.predict(X_train)
        
        # Criar DataFrame com dados originais, classes verdadeiras e clusters atribuídos
        if isinstance(y_train, pd.Series):
            y_values = y_train.values
        else:
            y_values = y_train
            
        df_clusters = pd.DataFrame(X_train, columns=features_selecionadas)
        df_clusters['cluster'] = clusters
        df_clusters['classe_verdadeira'] = y_values
        
        # Perfis de risco baseados na capacidade de pagamento e histórico
        perfis_risco = {
            'Baixo Risco': "Cliente com baixa razão de endividamento, alto histórico de pagamento e boa capacidade de pagamento",
            'Médio Risco': "Cliente com moderada razão de endividamento e capacidade de pagamento razoável",
            'Alto Risco': "Cliente com alta razão de endividamento e baixa capacidade de pagamento"
        }
        
        # Inicializar descrições
        descricoes = {}
        
        # Para cada cluster
        for cluster_id in range(model.n_clusters):
            # Filtrar dados do cluster
            cluster_data = df_clusters[df_clusters['cluster'] == cluster_id]
            
            # Obter estatísticas do cluster
            stats = cluster_data[features_selecionadas].describe()
            
            # Contar classes no cluster
            class_counts = cluster_data['classe_verdadeira'].value_counts(normalize=True) * 100
            
            # Determinar perfil de risco baseado em métricas
            razao_end_media = cluster_data['razao_endividamento'].mean() if 'razao_endividamento' in features_selecionadas else None
            capacidade_pag_media = cluster_data['capacidade_pagamento'].mean() if 'capacidade_pagamento' in features_selecionadas else None
            
            if razao_end_media is not None and capacidade_pag_media is not None:
                if razao_end_media < 0.3 and capacidade_pag_media > 3:
                    perfil = "Baixo Risco"
                elif razao_end_media > 0.7 or capacidade_pag_media < 1:
                    perfil = "Alto Risco"
                else:
                    perfil = "Médio Risco"
            else:
                perfil = "Não determinado"
            
            # Criar descrição do cluster
            classe_mapeada = mapeamento_clusters.get(cluster_id, "Desconhecida")
            
            # Mapeamento para nomes interpretáveis
            nome_classe = {
                1: "Não Elegível",
                2: "Elegível com Análise",
                3: "Elegível"
            }.get(classe_mapeada, str(classe_mapeada))
            
            descricao = {
                'id': cluster_id,
                'tamanho': len(cluster_data),
                'percentual': len(cluster_data) / len(df_clusters) * 100,
                'classe_mapeada': classe_mapeada,
                'nome_classe': nome_classe,
                'distribuicao_classes': class_counts.to_dict(),
                'estatisticas': stats,
                'perfil': perfil,
                'descricao_perfil': perfis_risco.get(perfil, "")
            }
            
            # Adicionar descrição à lista
            descricoes[cluster_id] = descricao
            
            # Logar informações
            logger.info(f"Cluster {cluster_id}:")
            logger.info(f"  - Tamanho: {descricao['tamanho']} ({descricao['percentual']:.1f}%)")
            logger.info(f"  - Classe mapeada: {classe_mapeada} ({nome_classe})")
            logger.info(f"  - Perfil: {perfil}")
            
            # Exibir informações
            print(f"\nCluster {cluster_id}:")
            print(f"  - Tamanho: {descricao['tamanho']} ({descricao['percentual']:.1f}%)")
            print(f"  - Classe mapeada: {classe_mapeada} ({nome_classe})")
            print(f"  - Perfil: {perfil}")
            print(f"  - Descrição: {descricao['descricao_perfil']}")
            
            # Exibir médias das principais features
            print("\n  Médias das features principais:")
            for feature in features_selecionadas:
                print(f"    - {feature}: {cluster_data[feature].mean():.4f}")
        
        return descricoes
    except Exception as e:
        logger.error(f"Erro ao descrever clusters: {str(e)}")
        return {}

def mapear_clusters_kmeans(model, X_train, y_train, retornar_detalhes=False):
    """
    Mapeia os clusters do K-Means para as classes reais com informações detalhadas
    
    Parâmetros:
    -----------
    model : KMeans
        Modelo K-Means treinado
    X_train : numpy.ndarray
        Dados de treinamento normalizados
    y_train : pandas.Series ou numpy.ndarray
        Rótulos verdadeiros
    retornar_detalhes : bool, opcional
        Se True, retorna também detalhes da distribuição de classes por cluster
        
    Retorna:
    --------
    dict ou tuple
        Mapeamento de cluster para classe predominante, e opcionalmente detalhes
    """
    try:
        logger.info("Mapeando clusters K-Means para classes")
        # Atribuir clusters aos dados de treinamento
        clusters = model.predict(X_train)
        
        # Criar mapeamento de cluster para classe
        mapping = {}
        detalhes = {}
        
        categorias = {1: 'Não Elegível', 2: 'Elegível c/ Análise', 3: 'Elegível'}
        
        for cluster_id in range(model.n_clusters):
            # Selecionar índices onde o cluster é igual a cluster_id
            indices = np.where(clusters == cluster_id)[0]
            
            # Contar as ocorrências de cada classe nesse cluster
            classes = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
            unique_classes, counts = np.unique(classes, return_counts=True)
            
            # A classe predominante é a que tem maior contagem
            predominant_class = unique_classes[np.argmax(counts)]
            mapping[cluster_id] = predominant_class
            
            # Calcular percentuais e salvar detalhes
            total = sum(counts)
            distribuicao = {}
            
            for i, classe in enumerate(unique_classes):
                percent = counts[i] / total * 100
                distribuicao[int(classe)] = {
                    'contagem': int(counts[i]),
                    'percentual': float(percent),
                    'nome': categorias.get(int(classe), str(classe))
                }
                
            detalhes[cluster_id] = {
                'tamanho_cluster': int(total),
                'classe_predominante': int(predominant_class),
                'nome_predominante': categorias.get(int(predominant_class), str(predominant_class)),
                'confianca': float(max(counts) / total * 100),
                'distribuicao': distribuicao
            }
            
            # Logging da distribuição
            logger.info(f"Cluster {cluster_id}: classe predominante {predominant_class} ({categorias.get(predominant_class, predominant_class)})")
            for i, classe in enumerate(unique_classes):
                percent = counts[i] / total * 100
                logger.info(f"  - Classe {classe} ({categorias.get(classe, classe)}): {counts[i]} ({percent:.2f}%)")
                
        if retornar_detalhes:
            return mapping, detalhes
        else:
            return mapping
    except Exception as e:
        logger.error(f"Erro ao mapear clusters K-Means: {str(e)}")
        if retornar_detalhes:
            return {}, {}
        else:
            return {}

def mapear_clusters_kmeans_ponderado(model, X_train, y_train, retornar_detalhes=False, thresholds=(0.4, 0.3, 0.0)):
    """
    Mapeia clusters para classes usando uma abordagem ponderada com thresholds
    para melhorar a distribuição das classes previstas
    
    Parâmetros:
    -----------
    model : KMeans
        Modelo K-Means treinado
    X_train : numpy.ndarray
        Dados de treinamento normalizados
    y_train : pandas.Series ou numpy.ndarray
        Rótulos verdadeiros
    retornar_detalhes : bool, opcional
        Se True, retorna também detalhes da distribuição de classes por cluster
    thresholds : tuple, opcional
        Thresholds para mapeamento (não_elegível, elegível_com_análise, elegível)
        
    Retorna:
    --------
    dict ou tuple
        Mapeamento de cluster para classe, e opcionalmente detalhes
    """
    try:
        logger.info("Mapeando clusters K-Means para classes usando abordagem ponderada")
        # Atribuir clusters aos dados de treinamento
        clusters = model.predict(X_train)
        
        # Criar mapeamento de cluster para classe
        mapping = {}
        detalhes = {}
        
        categorias = {1: 'Não Elegível', 2: 'Elegível c/ Análise', 3: 'Elegível'}
        
        for cluster_id in range(model.n_clusters):
            # Selecionar índices onde o cluster é igual a cluster_id
            indices = np.where(clusters == cluster_id)[0]
            
            # Contar as ocorrências de cada classe nesse cluster
            classes = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
            unique_classes, counts = np.unique(classes, return_counts=True)
            
            # Calcular distribuição percentual
            total = sum(counts)
            percentuais = {}
            for i, classe in enumerate(unique_classes):
                percentuais[int(classe)] = counts[i] / total
            
            # Definir classe usando thresholds
            if 1 in percentuais and percentuais[1] >= thresholds[0]:
                # Se a porcentagem de classe 1 (Não Elegível) for acima do threshold, mapeia para 1
                classe_escolhida = 1
            elif 2 in percentuais and percentuais[2] >= thresholds[1]:
                # Se a porcentagem de classe 2 (Elegível c/ Análise) for acima do threshold, mapeia para 2
                classe_escolhida = 2
            elif 3 in percentuais and percentuais[3] >= thresholds[2]:
                # Classe 3 (Elegível) com qualquer percentual acima do threshold, mapeia para 3
                classe_escolhida = 3
            else:
                # Caso nenhum threshold seja atingido, usa o método padrão (classe predominante)
                classe_escolhida = int(unique_classes[np.argmax(counts)])
            
            mapping[cluster_id] = classe_escolhida
            
            # Salvar detalhes para análise
            distribuicao = {}
            for i, classe in enumerate(unique_classes):
                percent = counts[i] / total * 100
                distribuicao[int(classe)] = {
                    'contagem': int(counts[i]),
                    'percentual': float(percent),
                    'nome': categorias.get(int(classe), str(classe))
                }
                
            detalhes[cluster_id] = {
                'tamanho_cluster': int(total),
                'classe_escolhida': int(classe_escolhida),
                'nome_classe': categorias.get(int(classe_escolhida), str(classe_escolhida)),
                'confianca': float(percentuais.get(classe_escolhida, 0) * 100),
                'distribuicao': distribuicao
            }
            
            # Logging da distribuição e classe escolhida
            logger.info(f"Cluster {cluster_id}: classe escolhida {classe_escolhida} ({categorias.get(classe_escolhida, classe_escolhida)})")
            for i, classe in enumerate(unique_classes):
                percent = counts[i] / total * 100
                logger.info(f"  - Classe {classe} ({categorias.get(classe, classe)}): {counts[i]} ({percent:.2f}%)")
                
        if retornar_detalhes:
            return mapping, detalhes
        else:
            return mapping
    except Exception as e:
        logger.error(f"Erro ao mapear clusters K-Means: {str(e)}")
        if retornar_detalhes:
            return {}, {}
        else:
            return {}

def salvar_modelo(model, scaler, caminho_modelo='models/model.joblib', caminho_scaler='models/scaler.joblib'):
    """
    Salva o modelo e o scaler em arquivos joblib
    """
    try:
        # Criar diretórios se não existirem
        Path(os.path.dirname(caminho_modelo)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(caminho_scaler)).mkdir(parents=True, exist_ok=True)
        
        # Salvar modelo e scaler
        dump(model, caminho_modelo)
        if scaler is not None:
            dump(scaler, caminho_scaler)
            logger.info(f"Scaler salvo em {caminho_scaler}")
        
        logger.info(f"Modelo salvo em {caminho_modelo}")
    except Exception as e:
        logger.error(f"Erro ao salvar modelo: {str(e)}")
        raise