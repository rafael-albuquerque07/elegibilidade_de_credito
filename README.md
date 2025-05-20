# 💳 Sistema de Elegibilidade de Crédito

![Badge](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Badge](https://img.shields.io/badge/Python-3.8%2B-blue)
![Badge](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Sobre o Projeto

Sistema de classificação de elegibilidade de crédito que avalia solicitações e as classifica em três categorias:

1. 🔴 **Não Elegível** - Solicitações que foram recusadas
2. 🟡 **Elegível com Análise** - Solicitações que necessitam de análise mais detalhada
3. 🟢 **Elegível** - Solicitações aprovadas imediatamente

Este projeto implementa uma solução completa para o problema, desde a engenharia de features até a implantação do modelo, utilizando algoritmos de aprendizado de máquina (K-Nearest Neighbors e K-Means Clustering).

## 🗂️ Estrutura do Projeto

```
elegibilidade_credito/
├── data/
│   ├── .gitkeep
│   └── elegibilidade_credito.csv
├── logs/
│   └── .gitkeep
├── models/
│   ├── .gitkeep
│   ├── model.joblib
│   ├── scaler.joblib
│   └── model_documentation.md
├── notebooks/
│   └── analise_elegibilidade_credito.ipynb
├── runs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   ├── model_utils.py
│   └── utils/
│       ├── __init__.py
│       ├── cli_utils.py
│       ├── logging_utils.py
│       └── tensorboard_utils.py
├── visualizacoes/
│   └── .gitkeep
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## 🔍 Características do Conjunto de Dados

O dataset contém informações sobre solicitações de crédito, incluindo:

- **Salário anual** do solicitante
- **Total de dívidas** já contraídas
- **Histórico de pagamento** (score de 0 a 1)
- **Idade** do solicitante
- **Crédito solicitado**
- **Elegibilidade** (1: Não elegível, 2: Elegível com análise, 3: Elegível)

## 🧠 Solução Implementada

### Modelos de Machine Learning

Foram implementados e comparados dois modelos:

1. **KNN (K-Nearest Neighbors)** - Modelo supervisionado que classifica com base na similaridade
2. **K-Means** - Modelo não supervisionado que agrupa solicitações similares

### Features Utilizadas

O conjunto de features selecionadas após análise:

```python
SELECTED_FEATURES = [
    'salario_anual',
    'total_dividas',
    'historico_pagamento',
    'razao_endividamento',
    'capacidade_pagamento'
]
```

### Features Derivadas (Engenharia de Features)

Foram criadas features derivadas para capturar melhor as relações financeiras:

1. **Razão de Endividamento** = `total_dividas / salario_anual`
2. **Capacidade de Pagamento** = `(salario_anual - total_dividas) / credito_solicitado`

Estas features derivadas incorporam conhecimento financeiro no modelo e melhoram significativamente seu desempenho.

### ⚠️ Decisão de Excluir a Feature "Idade"

Após análise detalhada, decidimos **não incluir a idade como feature** no modelo final. Esta decisão foi baseada nas seguintes evidências:

1. **Correlação insignificante com elegibilidade** (apenas 0.0073)
2. **Médias de idade quase idênticas entre as categorias**:
   - Não Elegível: 43.16 anos
   - Elegível c/ Análise: 43.08 anos
   - Elegível: 43.37 anos
3. **Distribuição uniforme por faixa etária** - A proporção de elegibilidade se mantém consistente em todas as faixas etárias
4. **Impacto negativo no desempenho** - A inclusão da idade reduziu a acurácia em 6.88%

Além disso, o próprio desafio sugere que a idade poderia ser ignorada se não fosse relevante.

🧠 Modelos Utilizados
🤖 KNN (K-Nearest Neighbors)

Algoritmo supervisionado
Melhor valor de K: 21 (escolhido via validação cruzada)
Acurácia: 84.90%

🔍 K-Means

Algoritmo não supervisionado
Clusters: 3
Acurácia após mapeamento: 78.02%
Métricas:

Calinski-Harabasz Score: 982.45 (maior é melhor)
Davies-Bouldin Score: 0.76 (menor é melhor)

### 🧩 Uso de Heurísticas

O projeto utiliza uma abordagem híbrida, combinando algoritmos de aprendizado de máquina com algumas heurísticas baseadas em conhecimento de domínio:

#### Heurísticas Aplicadas:

1. **Criação de features derivadas**: As fórmulas para razão de endividamento e capacidade de pagamento incorporam conhecimento financeiro tradicional

2. **Mapeamento de clusters**: Tradução de clusters K-Means para categorias de negócio através de uma heurística de classe predominante

3. **Perfis de risco**: Classificação de clusters em perfis de risco (baixo, médio, alto) com base em limiares financeiros predefinidos:
   ```python
   if razao_end_media < 0.3 and capacidade_pag_media > 3:
       perfil = "Baixo Risco"
   elif razao_end_media > 0.7 or capacidade_pag_media < 1:
       perfil = "Alto Risco"
   else:
       perfil = "Médio Risco"
   ```

A maior parte do processamento decisório, entretanto, é baseada em dados e não em heurísticas.

## 🚀 Melhorias Implementadas

O projeto vai muito além do solicitado no desafio, incluindo:

### 1. Arquitetura Modular e Profissional

- **Separação de responsabilidades** em módulos específicos
- **CLI robusta** para execução do pipeline
- **Sistema de logging** para rastreabilidade

### 2. Experimentação Avançada

- **Validação cruzada** para redução de overfitting
- **Testes sistemáticos** de diferentes valores de K
- **Métricas de cluster** (Calinski-Harabasz e Davies-Bouldin)

### 3. Visualizações e Análises Detalhadas

- **Matrizes de correlação** para entendimento de relações
- **Visualização PCA** para redução de dimensionalidade
- **Fronteiras de decisão** para interpretação do modelo

### 4. Recursos para Produção

- **Integração com TensorBoard** para monitoramento
- **Geração automática de documentação**
- **Tratamento robusto de erros**

### 5. Avaliação Aprofundada

- **Descrição interpretativa** dos clusters
- **Comparação automática** entre modelos
- **Relatórios em markdown e PDF**

## 📊 Resultados

### Comparação de Modelos

| Modelo        | Acurácia | Observações                |
| ------------- | -------- | -------------------------- |
| KNN (k=21)    | 84.90%   | Melhor desempenho geral    |
| K-Means (k=3) | 78.02%   | Solução não supervisionada |

### Métricas de Cluster (K-Means)

- **Calinski-Harabasz Score**: 982.45 (maior é melhor)
- **Davies-Bouldin Score**: 0.76 (menor é melhor)

## 🛠️ Como Usar

### Requisitos

```
scikit-learn>=0.24.0
pandas>=1.2.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
torch>=1.9.0  # Para TensorBoard
```

### Instalação

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/elegibilidade_credito.git
cd elegibilidade_credito

# Instalar dependências
pip install -r requirements.txt

# Criar estrutura de diretórios
mkdir -p data logs models notebooks runs visualizacoes
touch data/.gitkeep logs/.gitkeep models/.gitkeep runs/.gitkeep visualizacoes/.gitkeep
```

## 💻 Fluxo de Trabalho Recomendado: Notebook → main.py

Este projeto foi estruturado para seguir o fluxo de trabalho ideal em ciência de dados: exploração interativa primeiro, seguida de automação para produção.

### 1. Exploração com Notebook

Comece explorando os dados e modelos de forma interativa:

```bash
jupyter notebook notebooks/analise_elegibilidade_credito.ipynb
```

Nesta fase, você pode:

- Visualizar e analisar os dados em tempo real
- Experimentar diferentes valores de hiperparâmetros
- Avaliar visualmente o desempenho dos modelos
- Identificar a configuração ideal (ex: KNN com K=21)

### 2. Automação com main.py

Depois de identificar os melhores parâmetros no notebook, use-os com o script main.py:

```bash
# Executar com os parâmetros otimizados
python main.py --model knn --k 7 --no-tensorboard
```

## 📋 Exemplos de Uso do main.py

### Experimentação e Comparação de Modelos

```bash
# Executar validação cruzada com ambos os modelos
python main.py --model both --cross-validation --k-values 5,7,11,15,21,31,41,51

# Testar diferentes números de clusters para K-Means
python main.py --model kmeans --clusters 3 --random-state 42
python main.py --model kmeans --clusters 4 --random-state 42
python main.py --model kmeans --clusters 5 --random-state 42

# Experimentar sem normalização de dados
python main.py --model knn --k 51 --no-normalization
```

### Monitoramento e Logging Avançado

```bash
# Configuração com monitoramento detalhado via TensorBoard
python main.py --model both --runs experimentos/comparacao_maio/ --logs logs/detalhados/

# Salvar apenas o modelo, sem visualizações ou TensorBoard
python main.py --model knn --k 7 --no-visualizations --no-tensorboard --output models/apenas_modelo/
```

### Configuração para Ambientes de Produção

```bash
# Configuração para servidor de produção (caminhos absolutos)
python main.py --model knn --k 7 --output /var/models/credito/ --logs /var/log/credito/ --use-absolute-paths

# Modo silencioso para jobs agendados (crontab)
# para PowerShell
python main.py --model knn --k 7 --no-visualizations --no-tensorboard > $null

# para CMD
python main.py --model knn --k 7 --no-visualizations --no-tensorboard > NUL
```

### Análise de Robustez do Modelo

```bash
# Análise de sensibilidade com diferentes valores de test-size
python main.py --model knn --k 7 --test-size 0.1 --output models/test_10/
python main.py --model knn --k 7 --test-size 0.3 --output models/test_30/
python main.py --model knn --k 7 --test-size 0.5 --output models/test_50/

# Testar reprodutibilidade com diferentes seeds
python main.py --model knn --k 7 --random-state 42 --output models/seed_42/
python main.py --model knn --k 7 --random-state 101 --output models/seed_101/
```

### Processamento em Lote

```bash
# Script de lote para processar múltiplos arquivos
for arquivo in dados/*.csv; do
  nome=$(basename "$arquivo" .csv)
  echo "Processando $nome..."
  python main.py --data "$arquivo" --model knn --k 21 --output "modelos/$nome/"
done
```

### Grid Search de Hiperparâmetros

```bash
# Script para varredura de hiperparâmetros
for k in 5 11 15 21 31 41; do
    echo "Testando KNN com k=$k"
    python main.py --model knn --k $k --output models/grid_search/k_$k/ --no-visualizations
done
```

### Processamento por Lotes com Python

```python
# script_lotes.py
import subprocess
import os

# Configurações
modelos = ["knn", "kmeans"]
k_values = [5, 11, 21, 31]
datasets = ["clientes_varejo.csv", "clientes_premium.csv"]

# Loop de processamento
for dataset in datasets:
    dataset_base = os.path.splitext(dataset)[0]
    for modelo in modelos:
        if modelo == "knn":
            for k in k_values:
                output_dir = f"resultados/{dataset_base}/{modelo}_k{k}"
                cmd = [
                    "python", "main.py",
                    "--data", f"data/{dataset}",
                    "--model", modelo,
                    "--k", str(k),
                    "--output", output_dir
                ]
                print(f"Executando: {' '.join(cmd)}")
                subprocess.run(cmd)
        else:
            output_dir = f"resultados/{dataset_base}/{modelo}"
            cmd = [
                "python", "main.py",
                "--data", f"data/{dataset}",
                "--model", modelo,
                "--output", output_dir
            ]
            print(f"Executando: {' '.join(cmd)}")
            subprocess.run(cmd)
```

## 🔮 Predição com o Modelo Treinado

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
print(f"Predição: {predicao[0]}")
# Resultado: 3 (Elegível)
```

## 🌟 Benefícios da Abordagem Notebook → main.py

- **Melhor dos dois mundos:** exploração interativa + automação robusta
- **Reprodutibilidade:** parâmetros otimizados facilmente transferidos
- **Eficiência de trabalho:** ferramenta certa para cada fase do projeto
- **Escalabilidade:** migração natural de desenvolvimento para produção
- **Documentação viva:** notebook serve como registro do processo de desenvolvimento

## 📋 Conclusões

O sistema de elegibilidade de crédito desenvolvido consegue prever com alta acurácia a elegibilidade dos solicitantes, utilizando apenas características financeiras relevantes. A decisão de excluir a idade como feature demonstra a importância da análise de dados na tomada de decisões, evitando incluir variáveis que não contribuem (ou prejudicam) o desempenho do modelo.

A aplicação de algumas heurísticas combinadas com algoritmos de aprendizado de máquina fornece um equilíbrio entre conhecimento de domínio e descoberta de padrões nos dados, resultando em um modelo mais interpretável e eficaz.

As melhorias implementadas além do requisitado transformam este projeto em uma solução completa e profissional, pronta para ser utilizada em ambientes de produção.

## 📜 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests com melhorias, correções de bugs ou novas funcionalidades.
