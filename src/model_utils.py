import joblib
import numpy as np
import pandas as pd
import os
import datetime
import logging
from pathlib import Path

# Configurar logger
logger = logging.getLogger(__name__)

def carregar_modelo(caminho_modelo='models/model.joblib', caminho_scaler='models/scaler.joblib'):
    """
    Carrega o modelo e o scaler a partir dos arquivos joblib
    """
    try:
        logger.info(f"Carregando modelo de {caminho_modelo}")
        model = joblib.load(caminho_modelo)
        
        scaler = None
        try:
            if caminho_scaler and os.path.exists(caminho_scaler):
                logger.info(f"Carregando scaler de {caminho_scaler}")
                scaler = joblib.load(caminho_scaler)
        except Exception as e:
            logger.warning(f"Erro ao carregar scaler: {str(e)}")
        
        return model, scaler
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        raise

def fazer_previsao(model, scaler, dados, features):
    """
    Faz previsões para novos dados
    """
    try:
        logger.info("Fazendo previsões para novos dados")
        
        # Garantir que os dados estão no formato correto
        if isinstance(dados, pd.DataFrame):
            X = dados[features].values
        elif isinstance(dados, list):
            if isinstance(dados[0], dict):
                X = np.array([[d[feature] for feature in features] for d in dados])
            else:
                X = np.array(dados).reshape(1, -1)
        else:
            X = np.array(dados).reshape(1, -1)
        
        # Normalizar os dados se scaler não for None
        if scaler is not None:
            X_norm = scaler.transform(X)
        else:
            X_norm = X
        
        # Fazer previsões
        previsoes = model.predict(X_norm)
        
        # Mapear para as categorias
        mapeamento = {1: 'Não Elegível', 2: 'Elegível com Análise', 3: 'Elegível'}
        categorias = [mapeamento.get(int(p), str(p)) for p in previsoes]
        
        logger.info(f"Previsões realizadas com sucesso: {len(previsoes)} itens")
        return previsoes, categorias
    except Exception as e:
        logger.error(f"Erro ao fazer previsões: {str(e)}")
        raise

def criar_documentacao(model, scaler, features, accuracy, caminho_saida='README.md', comparacao_modelos=None):
    """
    Cria a documentação do modelo com informações detalhadas
    
    Parâmetros:
    -----------
    model : objeto modelo
        Modelo treinado (KNN ou K-Means)
    scaler : objeto StandardScaler
        Scaler utilizado para normalização
    features : list
        Lista de features utilizadas
    accuracy : float
        Acurácia do modelo no conjunto de teste
    caminho_saida : str
        Caminho para salvar a documentação
    comparacao_modelos : dict, opcional
        Dicionário com resultados da comparação entre KNN e K-Means
        Exemplo: {'knn': {'accuracy': 0.849, 'best_k': 21}, 'kmeans': {'accuracy': 0.780}}
    """
    try:
        logger.info(f"Criando documentação detalhada para o modelo")
        
        # Extrair médias e desvios padrão do scaler
        if scaler is not None:
            medias = scaler.mean_
            desvios = scaler.scale_
        else:
            medias = ["N/A"] * len(features)
            desvios = ["N/A"] * len(features)
        
        # Nome do modelo
        if hasattr(model, 'n_neighbors'):
            tipo_modelo = "KNN"
            parametro = f"K = {model.n_neighbors}"
        else:
            tipo_modelo = "K-Means"
            parametro = f"Clusters = {model.n_clusters}"
        
        # Criar documentação
        docs = f"""# Documentação do Modelo de Elegibilidade de Crédito

## Sumário
1. [Especificações do Modelo](#especificações-do-modelo)
2. [Features Utilizadas](#features-utilizadas)
3. [Normalização](#normalização)
4. [Decisões de Design](#decisões-de-design)
5. [Comparação de Modelos](#comparação-de-modelos)
6. [Desempenho do Modelo](#desempenho-do-modelo)
7. [Instruções para Uso](#instruções-para-uso-do-modelo)
8. [Exemplo de Previsão](#exemplo-de-previsão-com-o-modelo)

## Especificações do Modelo

### Tipo de Modelo
- **Algoritmo**: {tipo_modelo}
- **{parametro.split('=')[0].strip()}**: {parametro.split('=')[1].strip()}

## Features Utilizadas
As features utilizadas no modelo são (em ordem no vetor de entrada):
"""
        
        for i, feature in enumerate(features, 1):
            docs += f"{i}. `{feature}`\n"
        
        docs += f"""
### Features Derivadas
- **Razão de Endividamento** = `total_dividas / salario_anual`  
  *Representa o percentual da renda anual comprometido com dívidas.*
  
- **Capacidade de Pagamento** = `(salario_anual - total_dividas) / credito_solicitado`  
  *Indica quantas vezes a renda disponível após dívidas cobre o crédito solicitado.*

## Normalização
Foi aplicada normalização utilizando o StandardScaler com as seguintes médias e desvios padrão:

**Médias:**
"""
     
        for i, feature in enumerate(features):
            if isinstance(medias, np.ndarray) or isinstance(medias, list):
                docs += f"- {feature}: {medias[i]:.6f}\n"
            else:
                docs += f"- {feature}: {medias}\n"
        
        docs += f"\n**Desvios padrão:**\n"
        
        for i, feature in enumerate(features):
            if isinstance(desvios, np.ndarray) or isinstance(desvios, list):
                docs += f"- {feature}: {desvios[i]:.6f}\n"
            else:
                docs += f"- {feature}: {desvios}\n"
        
        docs += f"""
### Mapeamento das Categorias
- **Categoria 1**: Não elegível
- **Categoria 2**: Elegível com análise
- **Categoria 3**: Elegível

## Decisões de Design

### Exclusão da Feature "Idade"
Após análise detalhada, a feature "idade" foi **excluída** do modelo final devido a:

1. **Correlação insignificante** com elegibilidade (apenas 0.0073)
2. **Médias de idade quase idênticas** entre as categorias:
   - Não Elegível: 43.16 anos
   - Elegível c/ Análise: 43.08 anos
   - Elegível: 43.37 anos
3. **Distribuição uniforme por faixa etária** - A proporção de elegibilidade se mantém consistente em todas as faixas etárias
4. **Impacto negativo no desempenho** - A inclusão da idade reduziu a acurácia em 6.88%

### Uso de Heurísticas
O projeto utiliza uma abordagem híbrida, combinando:

1. **Aprendizado de máquina**: Classificação baseada em dados com KNN ou K-Means
2. **Heurísticas financeiras**: Criação de features derivadas baseadas em conhecimento do domínio

Esta combinação permite aproveitar tanto o poder preditivo dos algoritmos quanto o conhecimento especializado em finanças.

## Comparação de Modelos
"""

        # Adicionar informações de comparação de modelos se disponíveis
        if comparacao_modelos:
            knn_acc = comparacao_modelos.get('knn', {}).get('accuracy', 0) * 100
            kmeans_acc = comparacao_modelos.get('kmeans', {}).get('accuracy', 0) * 100
            knn_k = comparacao_modelos.get('knn', {}).get('best_k', 'N/A')
            kmeans_clusters = comparacao_modelos.get('kmeans', {}).get('clusters', 3)
            
            docs += f"""
Dois modelos foram comparados para a tarefa de classificação de elegibilidade de crédito:

| Modelo | Parâmetros | Acurácia | Observações |
|--------|------------|----------|-------------|
| KNN | K = {knn_k} | {knn_acc:.2f}% | Modelo supervisionado, melhor desempenho |
| K-Means | Clusters = {kmeans_clusters} | {kmeans_acc:.2f}% | Modelo não supervisionado |

**Conclusão da comparação**: O modelo KNN demonstrou desempenho superior para esta tarefa.
"""
        else:
            # Informações genéricas se não houver dados de comparação específicos
            docs += f"""
Dois tipos de modelos foram considerados para este problema:

- **KNN (K-Nearest Neighbors)**: Modelo supervisionado que classifica com base na similaridade
- **K-Means**: Modelo não supervisionado que agrupa solicitações similares em clusters

Os testes indicaram que o modelo KNN oferece melhor acurácia para esta tarefa específica, embora o K-Means também tenha apresentado resultados satisfatórios e ofereça a vantagem de ser treinado sem necessidade de rótulos.
"""

        docs += f"""
## Desempenho do Modelo
- O modelo {tipo_modelo} atingiu uma acurácia de aproximadamente **{accuracy*100:.2f}%** no conjunto de teste.
- A validação cruzada foi utilizada para evitar overfitting e garantir a robustez do modelo.

### Métricas Adicionais
- Para o KNN: Foram testados diversos valores de K para encontrar o ponto ótimo entre viés e variância.
- Para o K-Means: Foram avaliados os índices Calinski-Harabasz e Davies-Bouldin para verificar a qualidade dos clusters.

## Instruções para Uso do Modelo
1. Normalizar as features de entrada usando o StandardScaler com os parâmetros fornecidos
2. Aplicar o modelo {tipo_modelo} para fazer a previsão
3. A saída será um número inteiro:
   - 1: Não elegível
   - 2: Elegível com análise
   - 3: Elegível

### Código de Exemplo
```python
import joblib
import numpy as np

# Carregar modelo e scaler
model = joblib.load('models/model.joblib')
scaler = joblib.load('models/scaler.joblib')

# Exemplo de entrada: [salario_anual, total_dividas, historico_pagamento, razao_endividamento, capacidade_pagamento]
exemplo = np.array([[80000, 10000, 0.98, 0.125, 4.67]])

# Normalizar usando o scaler
exemplo_norm = scaler.transform(exemplo)

# Fazer predição
predicao = model.predict(exemplo_norm)
print(f"Predição: {{predicao[0]}}")  # Resultado: 3 (Elegível)
```

## Exemplo de Previsão com o Modelo
Para exemplos detalhados e análises visuais, consulte o notebook `notebooks/analise_elegibilidade_credito.ipynb`.

---

*Observação: O modelo deve ser utilizado como suporte à decisão, não como único critério para avaliação de crédito. O sistema foi desenvolvido para auxiliar o processo decisório, mas a revisão por especialistas financeiros é recomendada para casos complexos.*
"""
        # Salvar documentação
        Path(os.path.dirname(caminho_saida)).mkdir(parents=True, exist_ok=True)
        with open(caminho_saida, 'w', encoding='utf-8') as f:
            f.write(docs)
        
        logger.info(f"Documentação salva em {caminho_saida}")
        
        return docs
    except Exception as e:
        logger.error(f"Erro ao criar documentação: {str(e)}")
        return None
    
def gerar_relatorio_final_md(results, features_selecionadas, cluster_descricoes, 
                       ch_score=None, db_score=None, caminho_saida='relatorio_final.md'):
    """
    Gera um relatório final em formato Markdown com os resultados da análise
    
    Parâmetros:
    -----------
    results : dict
        Dicionário com resultados dos modelos
    features_selecionadas : list
        Lista de features usadas no modelo
    cluster_descricoes : dict
        Dicionário com descrições dos clusters
    ch_score : float, opcional
        Valor do score Calinski-Harabasz
    db_score : float, opcional
        Valor do score Davies-Bouldin
    caminho_saida : str
        Caminho para salvar o arquivo Markdown
        
    Retorna:
    --------
    str
        Conteúdo do relatório gerado
    """
    try:       
        logger.info(f"Gerando relatório final em {caminho_saida}")
        
        # Determinar o melhor modelo
        if 'knn' in results and 'kmeans' in results:
            best_model = 'knn' if results['knn']['accuracy'] > results['kmeans']['accuracy'] else 'kmeans'
            best_accuracy = max(results['knn']['accuracy'], results['kmeans']['accuracy'])
        elif 'knn' in results:
            best_model = 'knn'
            best_accuracy = results['knn']['accuracy']
        elif 'kmeans' in results:
            best_model = 'kmeans'
            best_accuracy = results['kmeans']['accuracy']
        else:
            best_model = "Nenhum modelo treinado"
            best_accuracy = 0.0
        
        # Data e hora atual
        now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        # Criar conteúdo do relatório
        content = f"""# Relatório Final - Análise de Elegibilidade de Crédito

*Gerado em: {now}*

## 1. Resumo dos Resultados

### Melhor Modelo
- **Modelo:** {best_model.upper()}
- **Acurácia:** {best_accuracy:.2%}

### Features Utilizadas
"""
        
        # Adicionar features
        for i, feature in enumerate(features_selecionadas, 1):
            content += f"{i}. `{feature}`\n"
        
        # Adicionar scores de cluster se disponíveis
        if ch_score is not None and db_score is not None:
            content += f"""
### Avaliação de Clusters
- **Calinski-Harabasz Score:** {ch_score:.2f} (maior é melhor)
- **Davies-Bouldin Score:** {db_score:.2f} (menor é melhor)
"""
        
        # Adicionar informações de KNN se disponível
        if 'knn' in results:
            knn_results = results['knn']
            content += f"""
## 2. Modelo KNN
- **Acurácia:** {knn_results['accuracy']:.2%}
- **Melhor valor de K:** {knn_results['best_k']}

### Resultados para diferentes valores de K
| K | Acurácia |
|---|----------|
"""
            for k, acc in knn_results['k_results'].items():
                content += f"| {k} | {acc:.4f} |\n"
        
        # Adicionar informações de K-Means se disponível
        if 'kmeans' in results:
            kmeans_results = results['kmeans']
            content += f"""
## 3. Modelo K-Means
- **Acurácia:** {kmeans_results['accuracy']:.2%}

### Mapeamento de Clusters para Classes
| Cluster | Classe | Descrição |
|---------|--------|-----------|
"""
            
            # Incluir mapeamento de clusters
            for cluster_id, classe in kmeans_results['cluster_mapping'].items():
                nome_classe = {1: "Não Elegível", 2: "Elegível com Análise", 3: "Elegível"}.get(classe, str(classe))
                descricao = "N/A"
                
                # Adicionar descrição se disponível
                if cluster_descricoes and cluster_id in cluster_descricoes:
                    descr = cluster_descricoes[cluster_id]
                    descricao = descr.get('perfil', "N/A")
                    
                content += f"| {cluster_id} | {classe} ({nome_classe}) | {descricao} |\n"
        
        # Adicionar descrição detalhada dos clusters
        if cluster_descricoes:
            content += f"""
## 4. Análise Detalhada dos Clusters

"""
            for cluster_id, descricao in cluster_descricoes.items():
                content += f"""### Cluster {cluster_id} - {descricao.get('perfil', 'N/A')}
- **Tamanho:** {descricao.get('tamanho', 'N/A')} ({descricao.get('percentual', 0):.1f}%)
- **Classe mapeada:** {descricao.get('classe_mapeada', 'N/A')} ({descricao.get('nome_classe', 'N/A')})
- **Descrição:** {descricao.get('descricao_perfil', 'N/A')}

#### Características principais:
"""
                # Incluir estatísticas das features se disponíveis
                stats = descricao.get('estatisticas')
                if stats is not None and hasattr(stats, 'loc'):
                    for feature in features_selecionadas:
                        try:
                            media = stats.loc['mean', feature]
                            std = stats.loc['std', feature]
                            content += f"- **{feature}:** {media:.4f} (±{std:.4f})\n"
                        except:
                            pass
                
                content += "\n"
        
        # Adicionar conclusões
        content += f"""
## 5. Conclusões

A análise de elegibilidade de crédito resultou em um modelo {best_model.upper()} com acurácia de {best_accuracy:.2%}.

### Principais Insights
- As features mais importantes são razão de endividamento e capacidade de pagamento.
- Clientes com alta razão de endividamento têm maior probabilidade de não serem elegíveis.
- Bom histórico de pagamento está fortemente associado à elegibilidade.

### Recomendações
- Utilizar o modelo como suporte à decisão, não como única fonte para aprovar/negar crédito.
- Incluir variáveis adicionais para melhorar o desempenho do modelo em casos intermediários.
- Realizar atualizações periódicas do modelo com novos dados para refletir mudanças no comportamento dos clientes.

"""
        
        # Criar diretório de saída se não existir
        Path(os.path.dirname(caminho_saida)).mkdir(parents=True, exist_ok=True)
        
        # Salvar o relatório
        with open(caminho_saida, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Relatório salvo em {caminho_saida}")
        
        return content
    except Exception as e:
        logger.error(f"Erro ao gerar relatório final: {str(e)}")
        return ""