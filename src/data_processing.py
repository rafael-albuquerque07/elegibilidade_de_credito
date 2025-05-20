import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import os
from pathlib import Path

# Configurar logger
logger = logging.getLogger(__name__)

def carregar_dados(caminho_arquivo):
    """
    Carrega dados do arquivo CSV e retorna um DataFrame
    """
    try:
        logger.info(f"Carregando dados de {caminho_arquivo}")
        df = pd.read_csv(caminho_arquivo)
        logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        raise

def processar_historico_pagamento(df):
    """
    Converte a coluna de histórico de pagamento para formato numérico
    """
    try:
        logger.info("Processando coluna de histórico de pagamento")
        # Fazer uma cópia para não alterar os dados originais
        df['historico_pagamento'] = df['historico_pagamento (score)'].copy()
        
        # Converter string para float
        if df['historico_pagamento'].dtype == 'object':
            df['historico_pagamento'] = df['historico_pagamento'].str.replace('.', '')
            df['historico_pagamento'] = df['historico_pagamento'].str.replace(',', '.')
            df['historico_pagamento'] = pd.to_numeric(df['historico_pagamento']) / 10000000000000000
            logger.info("Conversão de histórico de pagamento concluída")
        
        return df
    except Exception as e:
        logger.error(f"Erro ao processar histórico de pagamento: {str(e)}")
        raise

def criar_features_derivadas(df):
    """
    Cria features derivadas a partir das existentes
    """
    try:
        logger.info("Criando features derivadas")
        
        # Razão de endividamento
        df['razao_endividamento'] = df['total_dividas'] / df['salario_anual']
        
        # Capacidade de pagamento
        df['capacidade_pagamento'] = (df['salario_anual'] - df['total_dividas']) / df['credito_solicitado']
        
        logger.info("Features derivadas criadas com sucesso")
        return df
    except Exception as e:
        logger.error(f"Erro ao criar features derivadas: {str(e)}")
        raise

def normalizar_dados(X_treino, X_teste=None):
    """
    Normaliza os dados usando StandardScaler
    Se X_teste for fornecido, normaliza usando os parâmetros do X_treino
    """
    try:
        logger.info("Normalizando dados")
        scaler = StandardScaler()
        X_treino_norm = scaler.fit_transform(X_treino)
        
        if X_teste is not None:
            X_teste_norm = scaler.transform(X_teste)
            logger.info("Dados de treino e teste normalizados")
            return X_treino_norm, X_teste_norm, scaler
        
        logger.info("Dados normalizados")
        return X_treino_norm, scaler
    except Exception as e:
        logger.error(f"Erro ao normalizar dados: {str(e)}")
        raise

def preparar_dados(df, features_selecionadas, test_size=0.2, random_state=42, normalizar=True):
    """
    Prepara os dados para treinamento e teste
    """
    try:
        logger.info(f"Preparando dados com test_size={test_size} e random_state={random_state}")
        
        # Selecionar features e target
        X = df[features_selecionadas]
        y = df['elegibilidade']
        
        logger.info(f"Features selecionadas: {features_selecionadas}")
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Dados divididos: {X_train.shape[0]} amostras de treino e {X_test.shape[0]} amostras de teste")
        
        # Normalizar dados
        if normalizar:
            X_train_norm, X_test_norm, scaler = normalizar_dados(X_train, X_test)
        else:
            X_train_norm, X_test_norm = X_train.values, X_test.values
            scaler = None
            logger.info("Normalização desativada")
        
        return X_train_norm, X_test_norm, y_train, y_test, scaler
    except Exception as e:
        logger.error(f"Erro ao preparar dados: {str(e)}")
        raise