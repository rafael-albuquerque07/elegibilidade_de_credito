import os
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class MLSummaryWriter:
    """
    Wrapper para SummaryWriter do TensorBoard adaptado para projetos de ML
    """
    def __init__(self, experiment_name=None, log_dir='runs'):
        """
        Inicializa o SummaryWriter
        
        Parâmetros:
        -----------
        experiment_name : str, opcional
            Nome do experimento. Se None, usa um timestamp
        log_dir : str
            Diretório base para armazenar os logs
        """
        # Criar nome do experimento se não fornecido
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Criar diretório de logs
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar SummaryWriter
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
    def log_hyperparameters(self, hparams):
        """
        Registra hiperparâmetros no TensorBoard
        
        Parâmetros:
        -----------
        hparams : dict
            Dicionário de hiperparâmetros
        """
        # Converter valores para tipos simples para o TensorBoard
        hparams_dict = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                hparams_dict[key] = value
            else:
                hparams_dict[key] = str(value)
        
        self.writer.add_hparams(hparams_dict, {})
    
    def log_metrics(self, metrics, step=None):
        """
        Registra métricas no TensorBoard
        
        Parâmetros:
        -----------
        metrics : dict
            Dicionário de métricas
        step : int, opcional
            Passo ou iteração (se aplicável)
        """
        for metric_name, value in metrics.items():
            self.writer.add_scalar(metric_name, value, step)
    
    def log_confusion_matrix(self, cm, class_names, step=None):
        """
        Registra uma matriz de confusão como imagem no TensorBoard
        
        Parâmetros:
        -----------
        cm : numpy.ndarray
            Matriz de confusão
        class_names : list
            Lista com os nomes das classes
        step : int, opcional
            Passo ou iteração
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, 
                    yticklabels=class_names, cmap='Blues', ax=ax)
        plt.xlabel('Previsto')
        plt.ylabel('Verdadeiro')
        plt.title('Matriz de Confusão')
        
        # Converter figura para imagem
        self.writer.add_figure('Confusion Matrix', fig, step)
        plt.close(fig)
    
    def log_knn_accuracy_vs_k(self, k_values, accuracies, step=None):
        """
        Registra acurácia vs. K para o KNN
        
        Parâmetros:
        -----------
        k_values : list
            Lista de valores de K
        accuracies : list
            Lista de acurácias correspondentes
        step : int, opcional
            Passo ou iteração
        """
        # Adicionar como escalares para gráfico de linha
        for k, acc in zip(k_values, accuracies):
            self.writer.add_scalar('KNN/Accuracy_vs_K', acc, k)
        
        # Adicionar como texto formatado
        text = "| K | Acurácia |\n|---|---|\n"
        for k, acc in zip(k_values, accuracies):
            text += f"| {k} | {acc:.4f} |\n"
        self.writer.add_text('KNN/Accuracy_Table', text, step)
    
    def close(self):
        """
        Fecha o writer
        """
        self.writer.close()