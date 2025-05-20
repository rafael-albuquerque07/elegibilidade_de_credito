#!/usr/bin/env python3
"""
Elegibilidade de Crédito - Script Principal

Este script é o ponto de entrada para o CLI, permitindo executar o pipeline
completo de análise de elegibilidade de crédito a partir da linha de comando.
"""

import os
# Configuração para suprimir mensagens do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suprimir INFO e WARNING
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desativar operações oneDNN

import sys
import importlib
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import shutil
import subprocess

# Filtrar warnings
warnings.filterwarnings('ignore')

# Importar módulos do projeto
from src.utils.logging_utils import setup_logger, create_experiment_logger
from src.utils.cli_utils import parse_arguments
from src.utils.tensorboard_utils import MLSummaryWriter
import src.data_processing as data_processing
import src.model_training as model_training
import src.model_evaluation as model_evaluation
import src.model_utils as model_utils

def main():
    """
    Função principal do pipeline
    """
    try:
        # Processar argumentos
        args = parse_arguments()
        
        # Configurar logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(args.logs, f"elegibilidade_credito_{timestamp}.log")
        Path(args.logs).mkdir(parents=True, exist_ok=True)
        logger = setup_logger("elegibilidade_credito", log_file)
        
        # Registrar início da execução
        logger.info("Iniciando pipeline de Elegibilidade de Crédito")
        logger.info(f"Argumentos: {args}")
        
        # Configurar TensorBoard
        if not args.no_tensorboard:
            try:
                tensorboard = MLSummaryWriter(
                    experiment_name=f"elegibilidade_{timestamp}",
                    log_dir=args.runs
                )
                # Registrar hiperparâmetros
                tensorboard.log_hyperparameters({
                    'model': args.model,
                    'k': args.k,
                    'clusters': args.clusters,
                    'test_size': args.test_size,
                    'normalization': not args.no_normalization,
                    'cross_validation': args.cross_validation,
                    'random_state': args.random_state
                })
                logger.info(f"TensorBoard configurado em {tensorboard.log_dir}")
            except Exception as e:
                logger.warning(f"Erro ao configurar TensorBoard: {str(e)}")
                args.no_tensorboard = True
        
        # Carregar dados
        logger.info(f"Carregando dados de {args.data}")
        df = data_processing.carregar_dados(args.data)
        logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        # Processar dados
        logger.info("Processando dados")
        df = data_processing.processar_historico_pagamento(df)
        df = data_processing.criar_features_derivadas(df)
        logger.info("Dados processados")
        
        # Selecionar features
        features_selecionadas = [
            'salario_anual', 'total_dividas', 'historico_pagamento', 
            'razao_endividamento', 'capacidade_pagamento'
        ]
        logger.info(f"Features selecionadas: {features_selecionadas}")
        
        # Separar dados
        logger.info(f"Separando dados: test_size={args.test_size}, random_state={args.random_state}")
        X_train_norm, X_test_norm, y_train, y_test, scaler = data_processing.preparar_dados(
            df, features_selecionadas, test_size=args.test_size, 
            random_state=args.random_state, normalizar=not args.no_normalization
        )
        logger.info(f"Dados separados: {X_train_norm.shape[0]} treino, {X_test_norm.shape[0]} teste")
        
        # Treinar modelos
        results = {}
        
        # KNN
        if args.model in ['knn', 'both']:
            logger.info("Treinando modelo KNN")
            
            # Definir k
            k_values = [int(k) for k in args.k_values.split(',')]
            
            # Validação cruzada ou treino/teste simples
            if args.cross_validation:
                logger.info(f"Realizando validação cruzada para KNN com k={k_values}")
                resultados_knn, melhor_k = model_training.avaliar_knn_cross_validation(
                    X_train_norm, y_train, k_values
                )
            else:
                logger.info(f"Avaliando KNN com diferentes valores de k: {k_values}")
                resultados_knn = {}
                for k in k_values:
                    modelo = model_training.treinar_knn(X_train_norm, y_train, k)
                    acuracia, _ = model_evaluation.avaliar_modelo(modelo, X_test_norm, y_test)
                    resultados_knn[k] = acuracia
                    logger.info(f"KNN k={k}: acurácia={acuracia:.4f}")
                
                # Encontrar o melhor k
                melhor_k = max(resultados_knn, key=resultados_knn.get)
                logger.info(f"Melhor valor de k: {melhor_k} com acurácia: {resultados_knn[melhor_k]:.4f}")
            
            # TensorBoard para KNN
            if not args.no_tensorboard:
                acuracias = [resultados_knn[k] for k in k_values]
                tensorboard.log_knn_accuracy_vs_k(k_values, acuracias)
            
            # Treinar modelo final KNN
            k_final = args.k if args.k is not None else melhor_k
            modelo_knn = model_training.treinar_knn(X_train_norm, y_train, k_final)
            
            # Avaliar modelo final
            accuracy_knn, y_pred_knn = model_evaluation.avaliar_modelo(modelo_knn, X_test_norm, y_test)
            logger.info(f"Modelo KNN final (k={k_final}): acurácia={accuracy_knn:.4f}")
            
            # Salvar resultados
            results['knn'] = {
                'model': modelo_knn,
                'accuracy': accuracy_knn,
                'best_k': melhor_k,
                'k_results': resultados_knn,
                'predictions': y_pred_knn
            }
            
            # Matriz de confusão para TensorBoard
            if not args.no_tensorboard:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred_knn)
                tensorboard.log_confusion_matrix(
                    cm, ['Não Elegível', 'Elegível c/ Análise', 'Elegível']
                )
        
        # K-Means
        if args.model in ['kmeans', 'both']:
            logger.info(f"Treinando modelo K-Means com {args.clusters} clusters")
            modelo_kmeans = model_training.treinar_kmeans(
                X_train_norm, n_clusters=args.clusters, random_state=args.random_state
            )
            
            # Mapear clusters para classes usando a abordagem ponderada
            mapeamento_clusters = model_training.mapear_clusters_kmeans_ponderado(
                modelo_kmeans, X_train_norm, y_train, thresholds=(0.35, 0.25, 0.0)
            )
            
            # Fazer previsões
            clusters_teste = modelo_kmeans.predict(X_test_norm)
            y_pred_kmeans = np.array([mapeamento_clusters[c] for c in clusters_teste])
            
            # Calcular acurácia
            accuracy_kmeans = (y_pred_kmeans == y_test.values).mean()
            logger.info(f"Modelo K-Means: acurácia={accuracy_kmeans:.4f}")
            
            # Salvar resultados
            results['kmeans'] = {
                'model': modelo_kmeans,
                'accuracy': accuracy_kmeans,
                'cluster_mapping': mapeamento_clusters,
                'predictions': y_pred_kmeans
            }
            
            # Avaliação avançada de clusters (K-Means)
            ch_score, db_score = model_evaluation.avaliar_clusters_avancado(
                X_train_norm, modelo_kmeans.labels_
            )
            
            # Visualizar as métricas para o cluster único atual
            if not args.no_visualizations:
                model_evaluation.visualizar_metricas_unico_cluster(
                    ch_score, db_score,
                    salvar=True,
                    diretorio=args.visualizations,
                    usar_caminho_absoluto=args.use_absolute_paths
                )

            # Se solicitado, comparar diferentes números de clusters
            if args.compare_clusters:
                logger.info("Iniciando comparação de diferentes números de clusters")
                
                # Converter string para lista de inteiros
                cluster_range = [int(k) for k in args.cluster_range.split(',')]
                
                # Dicionário para armazenar métricas
                metricas_por_cluster = {}
                
                # Testar cada número de clusters
                for n_clusters in cluster_range:
                    logger.info(f"Testando K-Means com {n_clusters} clusters")
                    
                    # Treinar modelo com n_clusters
                    modelo_temp = model_training.treinar_kmeans(
                        X_train_norm, n_clusters=n_clusters, random_state=args.random_state
                    )
                    
                    # Avaliar clusters
                    ch_temp, db_temp = model_evaluation.avaliar_clusters_avancado(
                        X_train_norm, modelo_temp.labels_
                    )
                    
                    # Guardar métricas
                    metricas_por_cluster[f'k={n_clusters}'] = {'CH': ch_temp, 'DB': db_temp}
                
                # Visualizar métricas para diferentes números de clusters
                if not args.no_visualizations:
                    model_evaluation.visualizar_metricas_cluster(
                        metricas_por_cluster, 
                        salvar=True,
                        diretorio=args.visualizations,
                        usar_caminho_absoluto=args.use_absolute_paths
                    )
                
                # Registrar resultados para o relatório final
                if not args.no_tensorboard:
                    # Se TensorBoard estiver disponível, registrar resultados
                    try:
                        for config, metrics in metricas_por_cluster.items():
                            n_clusters = int(config.split('=')[1])
                            tensorboard.writer.add_scalars('Metrics/Calinski_Harabasz', 
                                                        {f'clusters_{n_clusters}': metrics['CH']}, 0)
                            tensorboard.writer.add_scalars('Metrics/Davies_Bouldin', 
                                                        {f'clusters_{n_clusters}': metrics['DB']}, 0)
                    except Exception as e:
                        logger.warning(f"Erro ao registrar métricas no TensorBoard: {str(e)}")
                
                # Identificar o melhor número de clusters com base nas métricas
                # Normalizar CH (maior é melhor) e DB (menor é melhor) para uma escala comum
                normalized_scores = {}
                
                # Extrair valores
                ch_values = [metricas_por_cluster[k]['CH'] for k in metricas_por_cluster]
                db_values = [metricas_por_cluster[k]['DB'] for k in metricas_por_cluster]
                
                # Normalizar CH (maior é melhor)
                ch_min, ch_max = min(ch_values), max(ch_values)
                ch_range = ch_max - ch_min if ch_max > ch_min else 1.0
                
                # Normalizar DB (menor é melhor)
                db_min, db_max = min(db_values), max(db_values)
                db_range = db_max - db_min if db_max > db_min else 1.0
                
                # Calcular pontuação combinada
                for k, metrics in metricas_por_cluster.items():
                    # Normalizar CH (0-1, maior é melhor)
                    ch_norm = (metrics['CH'] - ch_min) / ch_range if ch_range > 0 else 0.5
                    
                    # Normalizar DB (0-1, menor é melhor, então invertemos)
                    db_norm = 1.0 - ((metrics['DB'] - db_min) / db_range if db_range > 0 else 0.5)
                    
                    # Pontuação combinada (média simples)
                    normalized_scores[k] = (ch_norm + db_norm) / 2.0
                
                # Encontrar o melhor número de clusters
                best_k = max(normalized_scores, key=normalized_scores.get)
                best_score = normalized_scores[best_k]
                
                logger.info(f"Melhor número de clusters com base nas métricas: {best_k} (pontuação: {best_score:.4f})")
                logger.info("Comparação de clusters concluída")
                
                # Adicionar informações ao log
                print("\nResultados da comparação de clusters:")
                print(f"Melhor número de clusters: {best_k} (pontuação: {best_score:.4f})")
                print("\nMétricas por número de clusters:")
                for k, metrics in metricas_por_cluster.items():
                    print(f"  {k}: CH={metrics['CH']:.2f}, DB={metrics['DB']:.2f}, Score={normalized_scores[k]:.4f}")
                
                # Salvar resultados em arquivo CSV
                try:
                    results_df = pd.DataFrame([
                        {
                            'n_clusters': int(k.split('=')[1]),
                            'calinski_harabasz': metrics['CH'],
                            'davies_bouldin': metrics['DB'],
                            'combined_score': normalized_scores[k]
                        }
                        for k, metrics in metricas_por_cluster.items()
                    ])
                    
                    # Ordenar por número de clusters
                    results_df = results_df.sort_values('n_clusters')
                    
                    # Salvar CSV
                    csv_path = os.path.join(args.output, 'comparacao_clusters.csv')
                    results_df.to_csv(csv_path, index=False)
                    logger.info(f"Resultados da comparação de clusters salvos em {csv_path}")
                except Exception as e:
                    logger.warning(f"Erro ao salvar resultados da comparação de clusters: {str(e)}")
            
            # Obter tanto o mapeamento ponderado quanto os detalhes
            mapeamento_clusters, detalhes_mapeamento = model_training.mapear_clusters_kmeans_ponderado(
                modelo_kmeans, X_train_norm, y_train, thresholds=(0.35, 0.25, 0.0), retornar_detalhes=True
            )
            
            # Descrever clusters em detalhes
            descricoes_clusters = model_training.descrever_clusters_kmeans(
                modelo_kmeans, X_train_norm, y_train, df, features_selecionadas, mapeamento_clusters
            )
            
            # Exportar DataFrame com clusters atribuídos
            try:
                # Normalizar todo o conjunto de dados
                X_full = df[features_selecionadas].values
                X_full_norm = scaler.transform(X_full) if scaler is not None else X_full
                
                # Prever clusters para todas as amostras
                df_clusters = df.copy()
                df_clusters['cluster'] = modelo_kmeans.predict(X_full_norm)
                
                # Adicionar classe prevista e real
                df_clusters['classe_prevista'] = df_clusters['cluster'].map(mapeamento_clusters)
                df_clusters['classe_real'] = df['elegibilidade']  # Usar coluna original
                
                # Salvar para análise posterior
                caminho_csv = os.path.join(args.output, 'dados_com_clusters.csv')
                df_clusters.to_csv(caminho_csv, index=False)
                logger.info(f"DataFrame com clusters salvo em {caminho_csv}")
            except Exception as e:
                logger.warning(f"Erro ao exportar DataFrame com clusters: {str(e)}")
            
        # Selecionar o melhor modelo
        if 'knn' in results and 'kmeans' in results:
            best_model = 'knn' if results['knn']['accuracy'] > results['kmeans']['accuracy'] else 'kmeans'
            best_accuracy = max(results['knn']['accuracy'], results['kmeans']['accuracy'])
            logger.info(f"Melhor modelo: {best_model.upper()} com acurácia={best_accuracy:.4f}")
        elif 'knn' in results:
            best_model = 'knn'
            best_accuracy = results['knn']['accuracy']
        elif 'kmeans' in results:
            best_model = 'kmeans'
            best_accuracy = results['kmeans']['accuracy']
        else:
            logger.warning("Nenhum modelo foi treinado!")
            return 1
        
        # Salvar o melhor modelo
        Path(args.output).mkdir(parents=True, exist_ok=True)
        if best_model == 'knn':
            model_training.salvar_modelo(
                results['knn']['model'], scaler,
                caminho_modelo=os.path.join(args.output, 'model.joblib'),
                caminho_scaler=os.path.join(args.output, 'scaler.joblib')
            )
        else:
            model_training.salvar_modelo(
                results['kmeans']['model'], scaler,
                caminho_modelo=os.path.join(args.output, 'model.joblib'),
                caminho_scaler=os.path.join(args.output, 'scaler.joblib')
            )
        
        # Gerar relatório final
        if 'best_model' in locals():
            try:
                # Reunir resultados para o relatório
                report_results = {}
                
                if 'knn' in results:
                    report_results['knn'] = results['knn']
                
                if 'kmeans' in results:
                    report_results['kmeans'] = results['kmeans']
                
                # Valores de ch_score e db_score podem não estar disponíveis se o modelo não for K-Means
                ch_score_val = ch_score if 'ch_score' in locals() else None
                db_score_val = db_score if 'db_score' in locals() else None
                
                # Descrições de clusters podem não estar disponíveis se o modelo não for K-Means
                cluster_descriptions = descricoes_clusters if 'descricoes_clusters' in locals() else {}
                
                # Gerar relatório final em Markdown
                relatorio_md = model_utils.gerar_relatorio_final_md(
                    report_results,
                    features_selecionadas,
                    cluster_descriptions,
                    ch_score_val,
                    db_score_val,
                    os.path.join(args.output, 'relatorio_final.md')
                )
                
                # Opcionalmente, gerar PDF a partir do Markdown
                if hasattr(args, 'generate_pdf') and args.generate_pdf and shutil.which('pandoc'):
                    try:
                        pdf_path = os.path.join(args.output, 'relatorio_final.pdf')
                        cmd = f"pandoc {os.path.join(args.output, 'relatorio_final.md')} -o {pdf_path}"
                        subprocess.run(cmd, shell=True, check=True)
                        logger.info(f"Relatório PDF gerado em {pdf_path}")
                    except Exception as e:
                        logger.warning(f"Erro ao gerar PDF: {str(e)}")
                
                logger.info("Relatório final gerado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao gerar relatório final: {str(e)}")
        
        # Gerar documentação
        model_utils.criar_documentacao(
            results[best_model]['model'], 
            scaler, 
            features_selecionadas, 
            results[best_model]['accuracy'],
            os.path.join(args.output, 'model_documentation.md')
        )
        
        # Gerar relatório final
        logger.info("Pipeline concluído com sucesso!")
        logger.info(f"Melhor modelo: {best_model.upper()} com acurácia={best_accuracy:.4f}")
        logger.info(f"Documentação salva em {os.path.join(args.output, 'model_documentation.md')}")
        
        # Fechar TensorBoard
        if not args.no_tensorboard:
            tensorboard.close()
        
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Erro durante execução do pipeline: {str(e)}", exc_info=True)
        else:
            print(f"Erro durante execução do pipeline: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())