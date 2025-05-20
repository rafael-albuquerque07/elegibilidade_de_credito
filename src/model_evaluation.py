import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Configurar logger
logger = logging.getLogger(__name__)

def criar_diretorio_visualizacoes(caminho='../visualizacoes', usar_caminho_absoluto=False):
    """
    Cria o diretório para salvar as visualizações
    """
    try:
        if usar_caminho_absoluto:
            # Obter o caminho absoluto para o diretório raiz do projeto
            caminho_atual = os.path.abspath(os.path.dirname(__file__))
            
            # Se estamos em src/ ou notebooks/, subir um nível para chegar na raiz
            if os.path.basename(caminho_atual) in ['src', 'notebooks']:
                caminho_raiz = os.path.dirname(caminho_atual)
            else:
                # Já estamos na raiz ou em outro lugar
                caminho_raiz = caminho_atual
                
            # Remover '../' do caminho se presente
            if caminho.startswith('../'):
                caminho = caminho[3:]
                
            # Construir caminho absoluto
            caminho_final = os.path.join(caminho_raiz, caminho)
        else:
            # Usar o caminho relativo como está
            caminho_final = caminho
        
        # Criar o diretório
        os.makedirs(caminho_final, exist_ok=True)
        logger.info(f"Diretório de visualizações criado: {caminho_final}")
        return caminho_final
    except Exception as e:
        logger.error(f"Erro ao criar diretório de visualizações: {str(e)}")
        return caminho

def avaliar_modelo(model, X_test, y_test):
    """
    Avalia o modelo e retorna a acurácia
    """
    try:
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        logger.info(f"Avaliação do modelo: acurácia={accuracy:.4f}")
        return accuracy, y_pred
    except Exception as e:
        logger.error(f"Erro ao avaliar modelo: {str(e)}")
        raise

def plotar_distribuicao_classes(df, coluna='elegibilidade', 
                               titulos={1: 'Não Elegível', 2: 'Elegível c/ Análise', 3: 'Elegível'},
                               salvar=True, diretorio='../visualizacoes', 
                               usar_caminho_absoluto=True):
    """
    Plota a distribuição das classes como gráfico de barras
    """
    try:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=coluna, data=df, palette='viridis')
        plt.title('Distribuição das Classes de Elegibilidade', fontsize=14)
        plt.xlabel('Categoria de Elegibilidade', fontsize=12)
        plt.ylabel('Contagem', fontsize=12)
        
        # Adicionar rótulos às categorias
        plt.xticks(range(len(titulos)), titulos.values())
        
        # Adicionar contagens sobre as barras
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.1,
                    f'{int(height)}', ha="center")
            
            # Adicionar percentuais dentro das barras
            percentage = 100 * height / len(df)
            ax.text(p.get_x() + p.get_width()/2., height/2,
                    f'{percentage:.1f}%', ha="center", color="white", fontweight="bold")
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Salvar a visualização
        if salvar:
            diretorio_final = criar_diretorio_visualizacoes(diretorio, usar_caminho_absoluto)
            caminho_arquivo = os.path.join(diretorio_final, 'distribuicao_classes_elegibilidade.png')
            plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {caminho_arquivo}")
        
        plt.show()
    except Exception as e:
        logger.error(f"Erro ao plotar distribuição de classes: {str(e)}")

def plotar_matriz_confusao(y_true, y_pred, labels=None, 
                          salvar=True, diretorio='../visualizacoes', 
                          usar_caminho_absoluto=True,
                          nome_arquivo='matriz_confusao.png',  
                          titulo='Matriz de Confusão'): 
    """
    Plota a matriz de confusão como heatmap estilizado
    """
    try:
        if labels is None:
            labels = {1: 'Não Elegível', 2: 'Elegível c/ Análise', 3: 'Elegível'}
        label_values = list(labels.values())
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        
        # Normalizar a matriz de confusão
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Criar heatmap com percentuais
        ax = sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                        xticklabels=label_values, yticklabels=label_values,
                        annot_kws={"size": 14})
        
        # Adicionar percentuais
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.7, f'({cm_norm[i, j]:.1%})', 
                       ha='center', va='center', color='black' if cm_norm[i, j] < 0.7 else 'white')
        
        plt.xlabel('Previsto', fontsize=14)
        plt.ylabel('Verdadeiro', fontsize=14)
        plt.title(titulo, fontsize=16)
        plt.tight_layout()
        
        # Salvar a visualização
        if salvar:
            diretorio_final = criar_diretorio_visualizacoes(diretorio, usar_caminho_absoluto)
            caminho_arquivo = os.path.join(diretorio_final, nome_arquivo)
            plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {caminho_arquivo}")
        
        plt.show()
        
        report = classification_report(y_true, y_pred, target_names=label_values, output_dict=False)
        print("\nRelatório de classificação detalhado:")
        print(report)
        
    except Exception as e:
        logger.error(f"Erro ao plotar matriz de confusão: {str(e)}")

def plotar_boxplots_por_categoria(df, features, target='elegibilidade',
                                 salvar=True, diretorio='../visualizacoes',
                                 usar_caminho_absoluto=True):
    """
    Plota boxplots das features agrupadas por categoria de elegibilidade
    """
    try:
        plt.figure(figsize=(16, 12))
        
        for i, feature in enumerate(features, 1):
            plt.subplot(3, 2, i)
            sns.boxplot(x=target, y=feature, data=df, palette='viridis')
            plt.title(f'Distribuição de {feature} por Categoria', fontsize=12)
            plt.xlabel('Categoria de Elegibilidade', fontsize=10)
            plt.ylabel(feature, fontsize=10)
            plt.xticks([0, 1, 2], ['Não Elegível', 'Elegível c/ Análise', 'Elegível'])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Salvar a visualização
        if salvar:
            diretorio_final = criar_diretorio_visualizacoes(diretorio, usar_caminho_absoluto)
            caminho_arquivo = os.path.join(diretorio_final, 'boxplots_features_por_categoria.png')
            plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {caminho_arquivo}")
        
        plt.show()
    except Exception as e:
        logger.error(f"Erro ao plotar boxplots: {str(e)}")

def plotar_matriz_correlacao(df, features=None,
                            salvar=True, diretorio='../visualizacoes',
                            usar_caminho_absoluto=True):
    """
    Plota matriz de correlação como heatmap
    """
    try:
        if features is None:
            # Usar todas as colunas numéricas
            features = df.select_dtypes(include=['number']).columns.tolist()
        
        corr_matrix = df[features].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        ax = sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                        cmap=cmap, center=0, square=True, linewidths=.5,
                        cbar_kws={"shrink": .8})
        
        plt.title('Matriz de Correlação', fontsize=16)
        plt.tight_layout()
        
        # Salvar a visualização
        if salvar:
            diretorio_final = criar_diretorio_visualizacoes(diretorio, usar_caminho_absoluto)
            caminho_arquivo = os.path.join(diretorio_final, 'matriz_correlacao.png')
            plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {caminho_arquivo}")
        
        plt.show()
    except Exception as e:
        logger.error(f"Erro ao plotar matriz de correlação: {str(e)}")

def plotar_medias_por_categoria(df, features, target='elegibilidade',
                                salvar=True, diretorio='../visualizacoes',
                                usar_caminho_absoluto=True):
    """
    Plota gráfico de barras com as médias das features por categoria
    """
    try:
        # Calcular médias por categoria
        medias = df.groupby(target)[features].mean()
        
        plt.figure(figsize=(14, 8))
        medias.T.plot(kind='bar', colormap='viridis')
        plt.title('Média das Features por Categoria de Elegibilidade', fontsize=14)
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Valor Médio', fontsize=12)
        plt.legend(title='Categoria', labels=['Não Elegível', 'Elegível c/ Análise', 'Elegível'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Salvar a visualização
        if salvar:
            diretorio_final = criar_diretorio_visualizacoes(diretorio, usar_caminho_absoluto)
            caminho_arquivo = os.path.join(diretorio_final, 'medias_features_por_categoria.png')
            plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {caminho_arquivo}")
        
        plt.show()
        
        return medias
    except Exception as e:
        logger.error(f"Erro ao plotar médias por categoria: {str(e)}")
        return None

def plotar_scatter_derivadas(df, x='razao_endividamento', y='capacidade_pagamento', 
                            target='elegibilidade', salvar=True, diretorio='../visualizacoes', 
                            usar_caminho_absoluto=True):
    """
    Plota scatter plot das features derivadas coloridas por categoria
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Definir paleta de cores
        cores = {1: 'red', 2: 'blue', 3: 'green'}
        categorias = {1: 'Não Elegível', 2: 'Elegível c/ Análise', 3: 'Elegível'}
        
        # Plotar pontos por categoria
        for categoria in df[target].unique():
            subset = df[df[target] == categoria]
            plt.scatter(subset[x], subset[y], 
                       c=cores[categoria], alpha=0.6, 
                       label=categorias[categoria])
        
        plt.title(f'Relação entre {x} e {y} por Categoria', fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        
        # Salvar a visualização
        if salvar:
            diretorio_final = criar_diretorio_visualizacoes(diretorio, usar_caminho_absoluto)
            caminho_arquivo = os.path.join(diretorio_final, 'scatter_features_derivadas.png')
            plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {caminho_arquivo}")
        
        plt.show()
    except Exception as e:
        logger.error(f"Erro ao plotar scatter de features derivadas: {str(e)}")
    
def plotar_pca(X, y, n_components=2, salvar=True, diretorio='../visualizacoes',
              usar_caminho_absoluto=True):
    """
    Aplica PCA e plota visualização 2D dos dados
    """
    try:
        # Aplicar PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Criar DataFrame para facilitar a plotagem
        df_pca = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Categoria': y
        })
        
        # Plotar resultado do PCA
        plt.figure(figsize=(12, 8))
        
        categorias = {1: 'Não Elegível', 2: 'Elegível c/ Análise', 3: 'Elegível'}
        cores = {1: 'red', 2: 'blue', 3: 'green'}
        
        for categoria in np.unique(y):
            subset = df_pca[df_pca['Categoria'] == categoria]
            plt.scatter(subset['PC1'], subset['PC2'], 
                       c=cores[categoria], alpha=0.6, 
                       label=categorias[categoria])
        
        # Adicionar informações de variância explicada
        explained_variance = pca.explained_variance_ratio_
        plt.title(f'PCA - Visualização 2D (Variância explicada: {sum(explained_variance)*100:.2f}%)', 
                  fontsize=14)
        plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        
        # Salvar a visualização
        if salvar:
            diretorio_final = criar_diretorio_visualizacoes(diretorio, usar_caminho_absoluto)
            caminho_arquivo = os.path.join(diretorio_final, 'pca_visualizacao_2d.png')
            plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {caminho_arquivo}")
        
        plt.show()
        
        return pca
    except Exception as e:
        logger.error(f"Erro ao plotar PCA: {str(e)}")
        return None

def plotar_pca_variancia(X, y=None, n_components=None, threshold=0.95, 
                    salvar=True, diretorio='../visualizacoes',
                    usar_caminho_absoluto=True):
    """
    Aplica PCA e plota a variância explicada por componente,
    além de determinar quantos componentes são necessários para
    atingir um determinado threshold de variância explicada
    
    Parâmetros:
    -----------
    X : numpy.ndarray
        Dados de entrada (features)
    y : numpy.ndarray, opcional
        Rótulos das classes (se fornecidos, colore o gráfico)
    n_components : int, opcional
        Número máximo de componentes a considerar (default: min(n_samples, n_features))
    threshold : float, opcional
        Threshold de variância explicada acumulada (default: 0.95)
    salvar : bool, opcional
        Se True, salva as visualizações em arquivo
    diretorio : str, opcional
        Diretório para salvar as visualizações
    usar_caminho_absoluto : bool, opcional
        Se True, usa caminhos absolutos para salvar arquivos
        
    Retorna:
    --------
    tuple
        (pca, n_components_selected)
    """
    try:
        logger.info("Aplicando PCA para análise de variância explicada")
        
        # Configurar número máximo de componentes
        max_components = min(X.shape[0], X.shape[1]) if n_components is None else min(n_components, min(X.shape[0], X.shape[1]))
        
        # Aplicar PCA com todos os componentes possíveis
        pca = PCA(n_components=max_components)
        X_pca = pca.fit_transform(X)
        
        # Obter variância explicada
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Determinar número de componentes para threshold
        n_components_selected = np.argmax(cumulative_variance >= threshold) + 1
        
        # Criar figura para variância explicada por componente
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, len(explained_variance_ratio) + 1), 
                explained_variance_ratio, alpha=0.8, 
                label='Variância explicada individual')
        plt.step(range(1, len(cumulative_variance) + 1), 
                 cumulative_variance, where='mid', 
                 label='Variância explicada acumulada')
        plt.axhline(y=threshold, linestyle='--', color='r', 
                   label=f'Threshold ({threshold:.0%})')
        plt.axvline(x=n_components_selected, linestyle='--', color='g', 
                   label=f'Componentes necessários: {n_components_selected}')
        
        plt.xlabel('Número de Componentes', fontsize=12)
        plt.ylabel('Variância Explicada', fontsize=12)
        plt.title('Variância Explicada por Componente Principal', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Salvar a visualização
        if salvar:
            diretorio_final = criar_diretorio_visualizacoes(diretorio, usar_caminho_absoluto)
            caminho_arquivo = os.path.join(diretorio_final, 'pca_variancia_explicada.png')
            plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {caminho_arquivo}")
        
        plt.show()
        
        # Visualizar apenas com 2 componentes se y for fornecido
        if y is not None and len(np.unique(y)) > 1:
            # Criar DataFrame para plotagem
            df_pca = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Categoria': y
            })
            
            # Plotar scatter
            plt.figure(figsize=(12, 8))
            
            categorias = {1: 'Não Elegível', 2: 'Elegível c/ Análise', 3: 'Elegível'}
            cores = {1: 'red', 2: 'blue', 3: 'green'}
            
            for categoria in np.unique(y):
                subset = df_pca[df_pca['Categoria'] == categoria]
                plt.scatter(subset['PC1'], subset['PC2'], 
                          c=cores.get(categoria, 'gray'), alpha=0.6, 
                          label=categorias.get(categoria, f'Categoria {categoria}'))
            
            # Variância explicada pelos dois primeiros componentes
            var_pc1 = explained_variance_ratio[0] * 100
            var_pc2 = explained_variance_ratio[1] * 100
            
            plt.title(f'PCA - Visualização 2D (PC1: {var_pc1:.2f}%, PC2: {var_pc2:.2f}%)', fontsize=14)
            plt.xlabel(f'PC1 ({var_pc1:.2f}%)', fontsize=12)
            plt.ylabel(f'PC2 ({var_pc2:.2f}%)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Adicionar texto informativo
            plt.figtext(0.5, 0.01, 
                      f'Nota: {n_components_selected} componentes são necessários para explicar {threshold:.0%} da variância', 
                      ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            # Salvar a visualização
            if salvar:
                caminho_arquivo = os.path.join(diretorio_final, 'pca_visualizacao_2d_explicada.png')
                plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
                logger.info(f"Visualização 2D salva em: {caminho_arquivo}")
                
            plt.show()
        
        logger.info(f"PCA concluída. {n_components_selected} componentes para {threshold:.0%} da variância.")
        return pca, n_components_selected
    except Exception as e:
        logger.error(f"Erro ao executar PCA com variância explicada: {str(e)}")
        return None, 0

def plotar_fronteira_decisao(model, X, y, feature_names, feature_indices=[0, 1],
                           salvar=True, diretorio='../visualizacoes',
                           usar_caminho_absoluto=True):
    """
    Plota a fronteira de decisão do modelo usando as duas features mais importantes
    """
    try:
        # Selecionar as duas features para visualização
        X_subset = X[:, feature_indices]
        
        # Criar um modelo temporário apenas para visualização
        from sklearn.neighbors import KNeighborsClassifier
        if hasattr(model, 'n_neighbors'):  # Se for KNN
            temp_model = KNeighborsClassifier(n_neighbors=model.n_neighbors)
            temp_model.fit(X_subset, y)
            titulo = f'Fronteira de Decisão do Modelo KNN (k={model.n_neighbors})'
        else:  # Se for K-Means ou outro
            from sklearn.cluster import KMeans
            temp_model = KMeans(n_clusters=3, random_state=42)
            temp_model.fit(X_subset)
            titulo = 'Fronteira de Decisão do Modelo K-Means'
        
        # Criar uma malha para plotar a fronteira de decisão
        plt.figure(figsize=(12, 8))
        
        # Plotar a fronteira de decisão usando o modelo temporário
        DecisionBoundaryDisplay.from_estimator(
            temp_model, X_subset, cmap='viridis', alpha=0.8,
            xlabel=feature_names[feature_indices[0]],
            ylabel=feature_names[feature_indices[1]],
        )
        
        # Plotar os pontos de dados
        cores = {1: 'red', 2: 'blue', 3: 'green'}
        categorias = {1: 'Não Elegível', 2: 'Elegível c/ Análise', 3: 'Elegível'}
        
        for categoria in np.unique(y):
            mask = y == categoria
            plt.scatter(X_subset[mask, 0], X_subset[mask, 1], 
                       c=cores[categoria], alpha=0.6, 
                       label=categorias[categoria], edgecolors='k')
        
        plt.title(titulo, fontsize=14)
        plt.xlabel(feature_names[feature_indices[0]], fontsize=12)
        plt.ylabel(feature_names[feature_indices[1]], fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        
        # Salvar a visualização
        if salvar:
            diretorio_final = criar_diretorio_visualizacoes(diretorio, usar_caminho_absoluto)
            caminho_arquivo = os.path.join(diretorio_final, 'fronteira_decisao_modelo.png')
            plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {caminho_arquivo}")
        
        plt.show()
    except Exception as e:
        logger.error(f"Erro ao plotar fronteira de decisão: {str(e)}")

def plotar_acuracia_diferentes_k(k_values, acuracias, 
                                salvar=True, diretorio='../visualizacoes',
                                usar_caminho_absoluto=True):
    """
    Plota um gráfico da acurácia para diferentes valores de K no KNN
    """
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(k_values, acuracias, marker='o', linestyle='-', linewidth=2)
        
        # Destacar o melhor K
        melhor_k = k_values[np.argmax(acuracias)]
        melhor_acuracia = max(acuracias)
        
        plt.axvline(x=melhor_k, color='r', linestyle='--', 
                   label=f'Melhor k = {melhor_k}')
        plt.scatter(melhor_k, melhor_acuracia, color='red', s=100, zorder=10)
        
        plt.title('Acurácia do KNN para Diferentes Valores de k', fontsize=14)
        plt.xlabel('Valor de k', fontsize=12)
        plt.ylabel('Acurácia', fontsize=12)
        plt.xticks(k_values)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Adicionar anotação para o melhor valor
        plt.annotate(f'k={melhor_k}: {melhor_acuracia:.2%}', 
                    xy=(melhor_k, melhor_acuracia),
                    xytext=(melhor_k+2, melhor_acuracia-0.02),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    fontsize=10)
        
        plt.tight_layout()
        
        # Salvar a visualização
        if salvar:
            diretorio_final = criar_diretorio_visualizacoes(diretorio, usar_caminho_absoluto)
            caminho_arquivo = os.path.join(diretorio_final, 'acuracia_diferentes_k.png')
            plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {caminho_arquivo}")
        
        plt.show()
        
        return melhor_k, melhor_acuracia
    except Exception as e:
        logger.error(f"Erro ao plotar acurácia para diferentes k: {str(e)}")
        return k_values[np.argmax(acuracias)], max(acuracias)

def comparar_modelos(modelos, acuracias, salvar=True, diretorio='../visualizacoes',
                    usar_caminho_absoluto=True):
    """
    Plota um gráfico comparativo da acurácia de diferentes modelos
    """
    try:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(modelos, acuracias, color=['#1f77b4', '#ff7f0e'])
        plt.title('Comparação da Acurácia dos Modelos', fontsize=14)
        plt.xlabel('Modelo', fontsize=12)
        plt.ylabel('Acurácia', fontsize=12)
        plt.ylim([0, 1])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adicionar valores de acurácia acima das barras
        for bar, acuracia in zip(bars, acuracias):
            plt.text(bar.get_x() + bar.get_width()/2, acuracia + 0.01, 
                    f'{acuracia:.2%}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        
        # Salvar a visualização
        if salvar:
            diretorio_final = criar_diretorio_visualizacoes(diretorio, usar_caminho_absoluto)
            caminho_arquivo = os.path.join(diretorio_final, 'comparacao_modelos.png')
            plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização salva em: {caminho_arquivo}")
        
        plt.show()
    except Exception as e:
        logger.error(f"Erro ao plotar comparação de modelos: {str(e)}")

def avaliar_clusters_avancado(X, labels):
    """
    Avalia a qualidade dos clusters usando métricas avançadas
    
    Parâmetros:
    -----------
    X : numpy.ndarray
        Dados de entrada (features)
    labels : numpy.ndarray
        Rótulos dos clusters
        
    Retorna:
    --------
    tuple
        (calinski_harabasz_score, davies_bouldin_score)
    """
    try:
        ch_score = calinski_harabasz_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        
        logger.info(f"Avaliação avançada de clusters:")
        logger.info(f"  - Calinski-Harabasz: {ch_score:.2f} (maior é melhor)")
        logger.info(f"  - Davies-Bouldin: {db_score:.2f} (menor é melhor)")
        
        print("\nAvaliação avançada de clusters:")
        print(f"  - Calinski-Harabasz: {ch_score:.2f} (maior é melhor)")
        print(f"  - Davies-Bouldin: {db_score:.2f} (menor é melhor)")
        
        return ch_score, db_score
    except Exception as e:
        logger.error(f"Erro ao avaliar clusters: {str(e)}")
        return None, None
    
def visualizar_metricas_cluster(metricas_dict, salvar=True, diretorio='../visualizacoes',
                              usar_caminho_absoluto=True):
    """
    Gera visualizações para métricas de avaliação de clusters como
    Calinski-Harabasz e Davies-Bouldin para diferentes configurações.
    
    Parâmetros:
    -----------
    metricas_dict : dict
        Dicionário com configurações e seus respectivos scores
        Formato: {config_name: {'CH': ch_score, 'DB': db_score}}
    salvar : bool, opcional
        Se True, salva a visualização em arquivo
    diretorio : str, opcional
        Diretório para salvar a visualização
    usar_caminho_absoluto : bool, opcional
        Se True, usa caminhos absolutos para salvar arquivos
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        logger.info("Gerando visualização de métricas de cluster")
        
        # Extrair dados do dicionário
        configs = list(metricas_dict.keys())
        ch_scores = [metricas_dict[config]['CH'] for config in configs]
        db_scores = [metricas_dict[config]['DB'] for config in configs]
        
        # Criar figura com dois subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot para Calinski-Harabasz (maior é melhor)
        bars1 = ax1.bar(configs, ch_scores, color='skyblue')
        ax1.set_title('Calinski-Harabasz Score por Configuração', fontsize=14)
        ax1.set_xlabel('Configuração', fontsize=12)
        ax1.set_ylabel('Score (maior é melhor)', fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Destacar o melhor valor
        best_idx = np.argmax(ch_scores)
        bars1[best_idx].set_color('green')
        ax1.text(best_idx, ch_scores[best_idx], f'{ch_scores[best_idx]:.2f}', 
                ha='center', va='bottom', fontweight='bold')
        
        # Adicionar valores em cada barra
        for i, v in enumerate(ch_scores):
            if i != best_idx:  # Já adicionamos texto para o melhor
                ax1.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        # Plot para Davies-Bouldin (menor é melhor)
        bars2 = ax2.bar(configs, db_scores, color='salmon')
        ax2.set_title('Davies-Bouldin Score por Configuração', fontsize=14)
        ax2.set_xlabel('Configuração', fontsize=12)
        ax2.set_ylabel('Score (menor é melhor)', fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Destacar o melhor valor
        best_idx = np.argmin(db_scores)
        bars2[best_idx].set_color('green')
        ax2.text(best_idx, db_scores[best_idx], f'{db_scores[best_idx]:.2f}', 
                ha='center', va='bottom', fontweight='bold')
        
        # Adicionar valores em cada barra
        for i, v in enumerate(db_scores):
            if i != best_idx:  # Já adicionamos texto para o melhor
                ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        plt.suptitle('Métricas de Avaliação de Clusters', fontsize=16)
        plt.tight_layout()
        
        # Salvar a visualização
        if salvar:
            diretorio_final = criar_diretorio_visualizacoes(diretorio, usar_caminho_absoluto)
            caminho_arquivo = os.path.join(diretorio_final, 'metricas_clusters.png')
            plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização de métricas salva em: {caminho_arquivo}")
        
        plt.show()
        
        return fig
    except Exception as e:
        logger.error(f"Erro ao visualizar métricas de cluster: {str(e)}")
        return None

def visualizar_metricas_unico_cluster(ch_score, db_score, salvar=True, diretorio='../visualizacoes',
                                    usar_caminho_absoluto=True):
    """
    Gera visualização para métricas de avaliação de um único cluster.
    
    Parâmetros:
    -----------
    ch_score : float
        Valor do Calinski-Harabasz score
    db_score : float
        Valor do Davies-Bouldin score
    salvar : bool, opcional
        Se True, salva a visualização em arquivo
    diretorio : str, opcional
        Diretório para salvar a visualização
    usar_caminho_absoluto : bool, opcional
        Se True, usa caminhos absolutos para salvar arquivos
    """
    try:
        import matplotlib.pyplot as plt
        
        logger.info("Gerando visualização de métricas para cluster único")
        
        # Criar figura com gráfico de medidor para cada métrica
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Configuração de limites para os medidores (valores aproximados)
        ch_min, ch_max = 0, max(1000, ch_score * 1.5)  # Ajuste dinâmico para o valor real
        db_min, db_max = 0, max(2, db_score * 1.5)     # Ajuste dinâmico para o valor real
        
        # Cores para os medidores
        cmap_ch = plt.cm.RdYlGn  # Vermelho -> Amarelo -> Verde (para Calinski-Harabasz, maior é melhor)
        cmap_db = plt.cm.RdYlGn_r  # Verde -> Amarelo -> Vermelho (para Davies-Bouldin, menor é melhor)
        
        # Criar medidor para Calinski-Harabasz
        import numpy as np
        from matplotlib.patches import Wedge, Circle
        
        # Função auxiliar para criar medidor
        def create_gauge(ax, value, min_val, max_val, cmap, title, text):
            # Normalizar valor
            norm_value = (value - min_val) / (max_val - min_val)
            norm_value = max(0, min(norm_value, 1))  # Limitar entre 0 e 1
            
            # Criar arco de fundo
            theta1, theta2 = 140, 400  # Ângulos para o arco (220 graus de abertura)
            width = 0.2  # Largura do arco
            
            # Arco de fundo (cinza)
            background = Wedge(center=(0.5, 0), r=0.7, theta1=theta1, theta2=theta2, 
                              width=width, facecolor='lightgray', edgecolor='gray')
            ax.add_patch(background)
            
            # Arco de valor
            value_angle = theta1 + norm_value * (theta2 - theta1)
            foreground = Wedge(center=(0.5, 0), r=0.7, theta1=theta1, theta2=value_angle, 
                             width=width, facecolor=cmap(norm_value), edgecolor='white')
            ax.add_patch(foreground)
            
            # Círculo central
            center_circle = Circle(xy=(0.5, 0), radius=0.05, facecolor='white', edgecolor='gray')
            ax.add_patch(center_circle)
            
            # Textos
            ax.text(0.5, 0.25, title, ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(0.5, -0.05, text, ha='center', va='center', fontsize=12, color='black')
            
            # Adicionar marcas de escala
            n_ticks = 7
            for i in range(n_ticks):
                tick_norm = i / (n_ticks - 1)
                tick_angle = np.radians(theta1 + tick_norm * (theta2 - theta1))
                inner_r = 0.7 - width
                outer_r = 0.7 + 0.02
                
                x_inner = 0.5 + inner_r * np.cos(tick_angle)
                y_inner = 0 + inner_r * np.sin(tick_angle)
                x_outer = 0.5 + outer_r * np.cos(tick_angle)
                y_outer = 0 + outer_r * np.sin(tick_angle)
                
                ax.plot([x_inner, x_outer], [y_inner, y_outer], color='gray')
                
                # Adicionar valor da escala para algumas marcas
                if i % 2 == 0:
                    x_text = 0.5 + (outer_r + 0.05) * np.cos(tick_angle)
                    y_text = 0 + (outer_r + 0.05) * np.sin(tick_angle)
                    tick_value = min_val + tick_norm * (max_val - min_val)
                    ax.text(x_text, y_text, f'{tick_value:.0f}', ha='center', va='center', fontsize=8)
            
            # Configurar eixos
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.2, 1)
            ax.axis('off')
        
        # Criar medidores
        create_gauge(ax1, ch_score, ch_min, ch_max, cmap_ch, 'Calinski-Harabasz', 
                   f'Valor: {ch_score:.2f}\n(maior é melhor)')
        create_gauge(ax2, db_score, db_min, db_max, cmap_db, 'Davies-Bouldin', 
                   f'Valor: {db_score:.2f}\n(menor é melhor)')
        
        # Configurar figura
        plt.suptitle('Métricas de Avaliação do Modelo K-Means', fontsize=16)
        plt.tight_layout()
        
        # Adicionar interpretação
        plt.figtext(0.5, 0.01, 
                  'Interpretação: Calinski-Harabasz alto e Davies-Bouldin baixo indicam clusters bem formados.', 
                  ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Salvar a visualização
        if salvar:
            diretorio_final = criar_diretorio_visualizacoes(diretorio, usar_caminho_absoluto)
            caminho_arquivo = os.path.join(diretorio_final, 'metricas_cluster_unico.png')
            plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
            logger.info(f"Visualização de métricas para cluster único salva em: {caminho_arquivo}")
        
        plt.show()
        
        return fig
    except Exception as e:
        logger.error(f"Erro ao visualizar métricas para cluster único: {str(e)}")
        return None


def testar_exemplos(model, scaler, features):
    """
    Testa o modelo com exemplos predefinidos
    """
    try:
        categorias = {1: 'Não Elegível', 2: 'Elegível com Análise', 3: 'Elegível'}
        
        exemplos = [
            # Baixo risco
            {
                'salario_anual': 80000,
                'total_dividas': 10000,
                'historico_pagamento': 0.98,
                'razao_endividamento': 10000/80000,
                'capacidade_pagamento': (80000-10000)/15000,
                'descricao': 'Baixo Risco'
            },
            # Médio risco
            {
                'salario_anual': 55000,
                'total_dividas': 15000,
                'historico_pagamento': 0.85,
                'razao_endividamento': 15000/55000,
                'capacidade_pagamento': (55000-15000)/20000,
                'descricao': 'Médio Risco'
            },
            # Alto risco
            {
                'salario_anual': 30000,
                'total_dividas': 25000,
                'historico_pagamento': 0.65,
                'razao_endividamento': 25000/30000,
                'capacidade_pagamento': (30000-25000)/20000,
                'descricao': 'Alto Risco'
            }
        ]
        
        logger.info("Testando modelo com exemplos")
        resultados = []
        
        for exemplo in exemplos:
            # Preparar o exemplo como um array
            exemplo_array = np.array([exemplo[feature] for feature in features]).reshape(1, -1)
            
            # Normalizar o exemplo se o scaler não for None
            if scaler is not None:
                exemplo_norm = scaler.transform(exemplo_array)
            else:
                exemplo_norm = exemplo_array
            
            # Fazer previsão
            predicao = model.predict(exemplo_norm)[0]
            label = categorias.get(predicao, str(predicao))
            
            resultado = {
                'descricao': exemplo['descricao'],
                'salario_anual': exemplo['salario_anual'],
                'total_dividas': exemplo['total_dividas'],
                'historico_pagamento': exemplo['historico_pagamento'],
                'razao_endividamento': exemplo['razao_endividamento'],
                'capacidade_pagamento': exemplo['capacidade_pagamento'],
                'predicao': predicao,
                'label': label
            }
            resultados.append(resultado)
            
            logger.info(f"Exemplo {exemplo['descricao']}: Previsão = {predicao} ({label})")
        
        # Exibir resultados
        for r in resultados:
            print(f"\nExemplo - {r['descricao']}:")
            print(f"  Salário Anual: ${r['salario_anual']:,.2f}")
            print(f"  Total de Dívidas: ${r['total_dividas']:,.2f}")
            print(f"  Histórico de Pagamento: {r['historico_pagamento']*100:.2f}%")
            print(f"  Razão de Endividamento: {r['razao_endividamento']*100:.2f}%")
            print(f"  Capacidade de Pagamento: {r['capacidade_pagamento']:.2f}")
            print(f"  Predição: {r['predicao']} ({r['label']})")
        
        return resultados
    except Exception as e:
        logger.error(f"Erro ao testar exemplos: {str(e)}")
        return []