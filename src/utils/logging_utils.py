import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Configura e retorna um logger com o nome especificado.
    
    Parâmetros:
    -----------
    name : str
        Nome do logger
    log_file : str, opcional
        Caminho para o arquivo de log. Se None, não salva em arquivo.
    level : int
        Nível de logging (ex: logging.INFO, logging.DEBUG)
        
    Retorna:
    --------
    logging.Logger
        Logger configurado
    """
    # Criar diretório de logs se não existir
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configurar formato do log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configurar logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remover handlers existentes para evitar duplicação
    if logger.handlers:
        logger.handlers.clear()
    
    # Adicionar handler para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Adicionar handler para arquivo
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_timestamp():
    """
    Retorna um timestamp formatado para uso em nomes de arquivos
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_experiment_logger(experiment_name, log_dir="logs"):
    """
    Cria um logger específico para um experimento
    
    Parâmetros:
    -----------
    experiment_name : str
        Nome do experimento
    log_dir : str
        Diretório para armazenar os logs
        
    Retorna:
    --------
    logging.Logger
        Logger configurado para o experimento
    """
    # Criar nome do arquivo de log com timestamp
    timestamp = get_timestamp()
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_path = Path(log_dir) / log_filename
    
    # Configurar e retornar o logger
    return setup_logger(experiment_name, log_file=str(log_path))