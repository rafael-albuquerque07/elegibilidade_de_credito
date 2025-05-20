"""
Pacote de Elegibilidade de Crédito.

Este pacote contém módulos para processamento de dados, treinamento de modelos,
avaliação e utilidades para um sistema de classificação de elegibilidade de crédito.

Módulos:
- data_processing: Funções para carregar, limpar e preparar dados
- model_training: Funções para treinar modelos KNN e K-Means
- model_evaluation: Funções para avaliar o desempenho dos modelos
- model_utils: Utilitários gerais para usar os modelos treinados
- utils: Pacote de utilitários para logging, TensorBoard, etc.
"""

# Importar os módulos principais para facilitar o acesso
from . import data_processing
from . import model_training
from . import model_evaluation
from . import model_utils
from . import utils

# Definir a versão do pacote
__version__ = '1.0.0'

def init():
    """
    Inicializa recursos necessários para o pacote.
    Esta função pode ser expandida para configurar conexões,
    verificar dependências ou preparar recursos adicionais.
    """
    import logging
    logging.getLogger(__name__).info("Inicializando o pacote de Elegibilidade de Crédito...")
    # Adicione aqui qualquer inicialização necessária

# Se este arquivo for executado diretamente
if __name__ == "__main__":
    try:
        init()
        print("Pacote inicializado com sucesso.")
    except Exception as e:
        print(f"Erro durante a inicialização do pacote: {str(e)}")