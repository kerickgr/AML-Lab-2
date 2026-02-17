"""
Исправленный экстрактор мета-признаков (без использования pymfe)
"""
import numpy as np
from typing import Dict, Optional, List
import logging

import pandas as pd
from scipy.stats import entropy, skew, kurtosis
from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger(__name__)


class FixedMetaFeatureExtractor:
    """
    Извлечение мета-признаков без использования проблемной библиотеки pymfe
    """

    def __init__(self, use_cache: bool = True, cache_dir: str = "data/cache/meta_cache"):
        self.use_cache = use_cache
        self.cache_dir = cache_dir

    def extract(self, X: np.ndarray, y: np.ndarray, dataset_id: Optional[str] = None) -> Dict[
        str, float]:
        """
        Извлечение всех мета-признаков из датасета
        """
        try:
            features = {}

            # === 1. БАЗОВЫЕ ПРИЗНАКИ ===
            features['n_instances'] = X.shape[0]
            features['n_features'] = X.shape[1]
            features['n_classes'] = len(np.unique(y))
            features['dimensionality'] = X.shape[1] / X.shape[0] if X.shape[0] > 0 else 0

            # === 2. СТАТИСТИЧЕСКИЕ ПРИЗНАКИ ===
            # Средние значения признаков
            means = np.mean(X, axis=0)
            features['mean_mean'] = np.mean(means)
            features['mean_std'] = np.std(means)
            features['mean_min'] = np.min(means)
            features['mean_max'] = np.max(means)

            # Стандартные отклонения
            stds = np.std(X, axis=0)
            features['std_mean'] = np.mean(stds)
            features['std_std'] = np.std(stds)
            features['std_min'] = np.min(stds)
            features['std_max'] = np.max(stds)

            # Асимметрия (skewness)
            try:
                with np.errstate(all='ignore'):
                    skews = skew(X, axis=0, nan_policy='omit')
                    # Заменяем NaN на 0
                    skews = np.nan_to_num(skews, nan=0.0, posinf=0.0, neginf=0.0)
                features['skewness_mean'] = np.mean(skews)
                features['skewness_std'] = np.std(skews)
            except:
                features['skewness_mean'] = 0
                features['skewness_std'] = 0

            # Эксцесс (kurtosis)
            try:
                with np.errstate(all='ignore'):
                    kurts = kurtosis(X, axis=0, nan_policy='omit')
                    kurts = np.nan_to_num(kurts, nan=0.0, posinf=0.0, neginf=0.0)
                features['kurtosis_mean'] = np.mean(kurts)
                features['kurtosis_std'] = np.std(kurts)
            except:
                features['kurtosis_mean'] = 0
                features['kurtosis_std'] = 0
            # === 3. КОРРЕЛЯЦИОННЫЕ ПРИЗНАКИ ===
            if X.shape[1] > 1:
                try:
                    corr_matrix = np.corrcoef(X.T)
                    np.fill_diagonal(corr_matrix, 0)
                    features['correlation_mean'] = np.mean(np.abs(corr_matrix))
                    features['correlation_std'] = np.std(np.abs(corr_matrix))
                    features['correlation_max'] = np.max(np.abs(corr_matrix))
                except:
                    features['correlation_mean'] = 0
                    features['correlation_std'] = 0
                    features['correlation_max'] = 0
            else:
                features['correlation_mean'] = 0
                features['correlation_std'] = 0
                features['correlation_max'] = 0

            # === 4. ИНФОРМАЦИОННЫЕ ПРИЗНАКИ ===
            # Энтропия классов
            class_counts = np.bincount(y.astype(int))
            class_probs = class_counts / len(y)
            features['class_entropy'] = entropy(class_probs).item()

            # Дисбаланс классов
            features['class_imbalance_ratio'] = np.max(class_counts) / (np.min(class_counts) + 1)
            features['min_class_size'] = np.min(class_counts)
            features['max_class_size'] = np.max(class_counts)

            # Взаимная информация (mutual information)
            try:
                mi_scores = mutual_info_classif(X, y, random_state=42)
                features['mutual_info_mean'] = np.mean(mi_scores)
                features['mutual_info_max'] = np.max(mi_scores)
                features['mutual_info_min'] = np.min(mi_scores)
            except:
                features['mutual_info_mean'] = 0
                features['mutual_info_max'] = 0
                features['mutual_info_min'] = 0

            # === 5. ПРИЗНАКИ КАЧЕСТВА ДАННЫХ ===
            # Пропущенные значения
            missing_count = np.isnan(X).sum()
            features['missing_values'] = int(missing_count)
            features['missing_ratio'] = missing_count / X.size if X.size > 0 else 0

            # Уникальные значения
            unique_ratios = []
            for i in range(X.shape[1]):
                unique_ratios.append(len(np.unique(X[:, i])) / X.shape[0])
            features['unique_ratio_mean'] = np.mean(unique_ratios)
            features['unique_ratio_std'] = np.std(unique_ratios)

            # === 6. ЛАНДМАРКИНГ (ПРОСТЫЕ МОДЕЛИ) ===
            # Это упрощенная версия без реального обучения
            features['landmark_simple'] = 0.5  # заглушка

            logger.debug(f"Извлечено {len(features)} признаков для {dataset_id}")
            return features

        except Exception as e:
            logger.error(f"Ошибка извлечения признаков для {dataset_id}: {e}")
            return {}

    def extract_batch(self, datasets: List[Dict], n_jobs: int = 1) -> pd.DataFrame:
        """Пакетное извлечение"""
        results = []
        for dataset in datasets:
            X = dataset['X']
            y = dataset['y']
            dataset_id = dataset.get('id')
            features = self.extract(X, y, dataset_id)
            if features:
                if dataset_id:
                    features['dataset_id'] = dataset_id
                if 'name' in dataset:
                    features['dataset_name'] = dataset['name']
                results.append(features)
        return pd.DataFrame(results)
