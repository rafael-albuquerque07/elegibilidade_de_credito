import argparse
from pathlib import Path

def parse_arguments():
    """
    Processa argumentos da linha de comando usando argparse
    
    Retorna:
    --------
    argparse.Namespace
        Objeto com os argumentos processados
    """
    parser = argparse.ArgumentParser(
        description='Modelo de Elegibilidade de Crédito',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argumentos gerais
    parser.add_argument('--data', type=str, default='data/elegibilidade_credito.csv',
                        help='Caminho para o arquivo CSV de dados')
    parser.add_argument('--output', type=str, default='models',
                        help='Diretório para salvar os modelos treinados')
    parser.add_argument('--visualizations', type=str, default='visualizacoes',
                        help='Diretório para salvar as visualizações')
    parser.add_argument('--logs', type=str, default='logs',
                        help='Diretório para salvar os logs')
    parser.add_argument('--runs', type=str, default='runs',
                        help='Diretório para salvar os logs do TensorBoard')
    
    # Argumentos de configuração
    parser.add_argument('--model', type=str, choices=['knn', 'kmeans', 'both'], default='both',
                        help='Modelo a ser treinado')
    parser.add_argument('--k', type=int, default=None,
                        help='Valor de K para o KNN (se None, faz busca)')
    parser.add_argument('--k-values', type=str, default='5,7,11,15,21,31,41,51',
                        help='Valores de K para testar, separados por vírgula')
    parser.add_argument('--clusters', type=int, default=9,
                        help='Número de clusters para K-Means')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proporção dos dados para teste')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Seed para reprodutibilidade')
    
    # Flags de controle
    parser.add_argument('--no-normalization', action='store_true',
                        help='Desativar normalização de features')
    parser.add_argument('--no-visualizations', action='store_true',
                        help='Não gerar visualizações')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Não usar TensorBoard')
    parser.add_argument('--cross-validation', action='store_true',
                        help='Usar validação cruzada')
    parser.add_argument('--use-absolute-paths', action='store_true',
                        help='Usar caminhos absolutos para arquivos')
    parser.add_argument('--generate-pdf', action='store_true',
                        help='Gerar relatório em formato PDF (requer pandoc instalado)')
    parser.add_argument('--compare-clusters', action='store_true',
                        help='Comparar diferentes números de clusters usando métricas de qualidade')
    parser.add_argument('--cluster-range', type=str, default='2,3,4,5,6',
                        help='Valores de clusters para testar, separados por vírgula')
    
    return parser.parse_args()