#!/usr/bin/env python3
"""
Главный скрипт для запуска пайплайна мета-обучения
"""
import argparse
import sys
from pathlib import Path

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# Импортируем напрямую, без использования __init__.py
from src.data.collector import OpenMLCollector
from src.meta_features.fixed_extractor import FixedMetaFeatureExtractor
from src.algorithms.base_learners import BaseLearners
from src.algorithms.evaluation import AlgorithmEvaluator
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Meta-learning pipeline')

    parser.add_argument('--step', type=str, choices=[
        'all', 'collect', 'extract', 'evaluate', 'analyze', 'noise', 'ablation'
    ], default='all', help='Pipeline step to run')

    parser.add_argument('--limit', type=int, default=350,
                        help='Limit number of datasets')

    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching')

    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    return parser.parse_args()


def step_collect_data(args):
    """Шаг 1: Сбор данных с OpenML"""
    logger.info("=" * 60)
    logger.info("ШАГ 1: Сбор данных с OpenML")
    logger.info("=" * 60)

    collector = OpenMLCollector(
        cache_dir='data/cache/openml_cache',
        use_cache=not args.no_cache
    )

    datasets_df = collector.fetch_datasets_list(
        min_instances=100,
        max_instances=10000,
        min_features=3,
        max_features=500,
        min_classes=2,
        limit=args.limit
    )

    # Сохраняем список
    datasets_df.to_csv('data/raw/dataset_list.csv', index=False)
    logger.info(f"Сохранено {len(datasets_df)} датасетов")

    return datasets_df


def step_extract_features(datasets_df, args):
    """Шаг 2: Извлечение мета-признаков"""
    logger.info("=" * 60)
    logger.info("ШАГ 2: Извлечение мета-признаков")
    logger.info("=" * 60)

    # Используем FixedMetaFeatureExtractor напрямую
    extractor = FixedMetaFeatureExtractor(use_cache=not args.no_cache)
    collector = OpenMLCollector(use_cache=not args.no_cache)

    meta_features = []
    successful = 0
    failed = 0
    skipped_small = 0
    skipped_classes = 0
    skipped_no_numeric = 0

    for idx, row in tqdm(datasets_df.iterrows(), total=len(datasets_df),
                         desc="Извлечение мета-признаков"):
        dataset_id = row['did']

        try:
            X, y = collector.download_dataset(dataset_id)

            if X is None or y is None:
                failed += 1
                continue

            # Строгая фильтрация
            if X.shape[0] < 50:
                logger.debug(f"Датасет {dataset_id} слишком мал ({X.shape[0]} объектов)")
                skipped_small += 1
                continue

            if len(np.unique(y)) < 2:
                logger.debug(f"Датасет {dataset_id} имеет меньше 2 классов")
                skipped_classes += 1
                continue

            # Проверка на NaN
            if np.isnan(X).any():
                mask = ~np.isnan(X).any(axis=1)
                X = X[mask]
                y = y[mask]
                if X.shape[0] < 50:
                    skipped_small += 1
                    continue

            features = extractor.extract(X, y, dataset_id=str(dataset_id))

            if features:
                features['dataset_id'] = dataset_id
                features['dataset_name'] = row['name']
                meta_features.append(features)
                successful += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(f"Ошибка в датасете {dataset_id}: {e}")
            failed += 1

    df = pd.DataFrame(meta_features)
    df.to_csv('data/processed/meta_features_raw.csv', index=False)

    logger.info(f"Сохранены мета-признаки для {len(df)} датасетов")
    logger.info(f"Успешно: {successful}")
    logger.info(f"Пропущено (мало объектов): {skipped_small}")
    logger.info(f"Пропущено (нет данных): {skipped_no_numeric}")
    logger.info(f"Пропущено (мало классов): {skipped_classes}")
    logger.info(f"Ошибок: {failed}")

    return df


def step_evaluate_algorithms(datasets_df, args):
    """Шаги 3-4: Оценка алгоритмов"""
    logger.info("=" * 60)
    logger.info("ШАГ 3-4: Оценка алгоритмов")
    logger.info("=" * 60)

    collector = OpenMLCollector(use_cache=not args.no_cache)

    # Инициализируем алгоритмы
    learners = BaseLearners(['LogisticRegression', 'RandomForest', 'KNN'])
    evaluator = AlgorithmEvaluator(
        learners.get_all_algorithms(),
        cv_folds=5,
        scoring='balanced_accuracy'
    )

    results = []
    successful = 0
    failed = 0

    for idx, row in tqdm(datasets_df.iterrows(), total=len(datasets_df),
                         desc="Оценка алгоритмов"):
        dataset_id = row['did']

        try:
            X, y = collector.download_dataset(dataset_id)

            if X is None or y is None:
                failed += 1
                continue

            # Проверки
            if X.shape[0] < 20:
                failed += 1
                continue

            if len(np.unique(y)) < 2:
                failed += 1
                continue

            # Оценка
            scores = evaluator.evaluate_all(X, y)

            if scores:
                result = {
                    'dataset_id': dataset_id,
                    'dataset_name': row['name'],
                    'best_algorithm': max(scores, key=scores.get),
                    **{f'score_{k}': v for k, v in scores.items()}
                }
                results.append(result)
                successful += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(f"Ошибка в датасете {dataset_id}: {e}")
            failed += 1

    df = pd.DataFrame(results)
    df.to_csv('data/processed/algorithm_performance.csv', index=False)
    logger.info(f"Оценено {len(df)} датасетов")
    logger.info(f"Успешно: {successful}, Ошибок: {failed}")

    return df


# ============== ЭТА ФУНКЦИЯ - ВСТАВЬТЕ ЕЕ СЮДА ==============
def step_analyze_meta(meta_df, results_df, args):
    """Шаги 5-6: Анализ мета-набора"""
    logger.info("=" * 60)
    logger.info("ШАГ 5-6: Анализ мета-набора")
    logger.info("=" * 60)

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    # Создаем папку для визуализаций
    import os
    os.makedirs('results/visualizations', exist_ok=True)

    # Объединяем мета-признаки с результатами
    meta_dataset = pd.merge(
        meta_df,
        results_df[['dataset_id', 'best_algorithm']],
        on='dataset_id'
    )

    # Сохраняем мета-набор
    meta_dataset.to_csv('results/meta_dataset/meta_dataset_complete.csv', index=False)
    logger.info(f"Мета-набор сохранен: {meta_dataset.shape}")

    # СОЗДАЕМ ВИЗУАЛИЗАЦИИ
    logger.info("Создание визуализаций...")

    # Определяем колонки с признаками
    feature_cols = [c for c in meta_dataset.columns
                    if c not in ['dataset_id', 'dataset_name', 'best_algorithm']]

    # 1. Корреляционная матрица
    plt.figure(figsize=(14, 12))
    corr = meta_dataset[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5)
    plt.title('Корреляционная матрица мета-признаков')
    plt.tight_layout()
    plt.savefig('results/visualizations/correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    # 2. PCA проекция
    X = meta_dataset[feature_cols].fillna(0).values
    y = meta_dataset['best_algorithm'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(12, 8))
    unique_algos = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_algos)))

    for i, algo in enumerate(unique_algos):
        mask = y == algo
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=algo, color=colors[i], alpha=0.7, s=50)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA проекция мета-набора')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/visualizations/pca_projection.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    # 3. t-SNE проекция
    # Проверяем, достаточно ли данных для t-SNE
    if len(X_scaled) > 10:
        tsne = TSNE(n_components=2, perplexity=min(30, len(X_scaled) - 1), random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)

        plt.figure(figsize=(12, 8))
        for i, algo in enumerate(unique_algos):
            mask = y == algo
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                        label=algo, color=colors[i], alpha=0.7, s=50)

        plt.xlabel('t-SNE1')
        plt.ylabel('t-SNE2')
        plt.title('t-SNE проекция мета-набора')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/visualizations/tsne_projection.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        logger.warning("Недостаточно данных для t-SNE (нужно > 10 точек)")

    # 4. Распределение лучших алгоритмов
    plt.figure(figsize=(10, 6))
    value_counts = meta_dataset['best_algorithm'].value_counts()
    value_counts.plot(kind='bar', color='coral', alpha=0.7, edgecolor='black')
    plt.title('Распределение лучших алгоритмов')
    plt.xlabel('Алгоритм')
    plt.ylabel('Количество датасетов')

    # Добавляем значения на столбцы
    for i, v in enumerate(value_counts.values):
        plt.text(i, v + 0.5, str(v), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/visualizations/best_algorithm_distribution.png', dpi=150,
                bbox_inches='tight')
    plt.show()
    plt.close()

    logger.info("Все визуализации сохранены в results/visualizations/")

    return meta_dataset


# ============== КОНЕЦ ФУНКЦИИ ==============

def step_noise_experiment(meta_dataset, args):
    """Шаг 9: Эксперимент с шумом"""
    logger.info("=" * 60)
    logger.info("ШАГ 9: Эксперимент с шумом")
    logger.info("=" * 60)

    # Здесь код для эксперимента с шумом
    # (можно добавить позже)

    return None


def main():
    """Главная функция"""
    args = parse_args()

    setup_logging(log_file='logs/meta_learning.log',
                  level='DEBUG' if args.debug else 'INFO')

    # Создаем директории
    for dir_path in ['data/raw', 'data/processed', 'data/cache',
                     'results/meta_dataset', 'results/visualizations',
                     'results/experiments', 'logs']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Запускаем шаги
    if args.step in ['all', 'collect']:
        datasets_df = step_collect_data(args)
    else:
        datasets_df = pd.read_csv('data/raw/dataset_list.csv')
        if args.limit:
            datasets_df = datasets_df.head(args.limit)

    if args.step in ['all', 'extract']:
        meta_df = step_extract_features(datasets_df, args)
    else:
        try:
            meta_df = pd.read_csv('data/processed/meta_features_raw.csv')
        except:
            meta_df = pd.DataFrame()

    if args.step in ['all', 'evaluate']:
        results_df = step_evaluate_algorithms(datasets_df, args)
    else:
        try:
            results_df = pd.read_csv('data/processed/algorithm_performance.csv')
        except:
            results_df = pd.DataFrame()

    if args.step in ['all', 'analyze']:
        if not meta_df.empty and not results_df.empty:
            meta_dataset = step_analyze_meta(meta_df, results_df, args)
        else:
            logger.error("Нет данных для анализа. Сначала выполните extract и evaluate.")

    if args.step in ['all', 'noise']:
        if 'meta_dataset' in locals():
            step_noise_experiment(meta_dataset, args)

    logger.info("=" * 60)
    logger.info("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
