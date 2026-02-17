import openml
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class OpenMLCollector:
    """Сбор датасетов с OpenML"""

    def __init__(self, cache_dir: str = "data/cache/openml_cache", use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

        openml.config.cache_directory = str(self.cache_dir)

    def fetch_datasets_list(self, **filters) -> pd.DataFrame:
        """Получение списка датасетов"""
        datasets_df = openml.datasets.list_datasets(output_format='dataframe')

        filtered = datasets_df.copy()

        if 'min_instances' in filters:
            filtered = filtered[filtered['NumberOfInstances'] >= filters['min_instances']]
        if 'max_instances' in filters:
            filtered = filtered[filtered['NumberOfInstances'] <= filters['max_instances']]
        if 'min_features' in filters:
            filtered = filtered[filtered['NumberOfFeatures'] >= filters['min_features']]
        if 'max_features' in filters:
            filtered = filtered[filtered['NumberOfFeatures'] <= filters['max_features']]
        if 'min_classes' in filters:
            filtered = filtered[filtered['NumberOfClasses'] >= filters['min_classes']]
        if 'limit' in filters:
            filtered = filtered.head(filters['limit'])

        logger.info(f"Найдено {len(filtered)} датасетов")
        return filtered

    def download_dataset(self, dataset_id: int) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]]:
        """Скачивание конкретного датасета"""
        logger.debug(f"Загрузка датасета {dataset_id}...")

        try:
            # Загружаем датасет с явными параметрами для подавления предупреждений
            dataset = openml.datasets.get_dataset(
                dataset_id,
                download_data=True,
                download_qualities=True,
                download_features_meta_data=True
            )

            # Получаем данные в формате dataframe (новый рекомендуемый формат)
            X, y, categorical, _ = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format='dataframe'  # Используем dataframe вместо array
            )

            # Конвертируем в numpy с обработкой строк
            try:
                # Пробуем конвертировать напрямую
                X = X.to_numpy(dtype=np.float64)
            except:
                # Если есть строки, пытаемся конвертировать только числовые колонки
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    X = X[numeric_cols].to_numpy(dtype=np.float64)
                else:
                    # Если нет числовых колонок, пропускаем датасет
                    logger.warning(f"Датасет {dataset_id} не содержит числовых признаков")
                    return None, None

            # Конвертируем y
            try:
                y = y.to_numpy(dtype=np.int64)
            except:
                # Если y не числовой, кодируем
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Удаляем строки с NaN
            mask = ~np.isnan(X).any(axis=1)
            X = X[mask]
            y = y[mask]

            # Проверка на пустой датасет
            if X.shape[0] == 0:
                logger.warning(f"Датасет {dataset_id} пуст после очистки")
                return None, None

            logger.debug(f"Датасет {dataset_id} загружен: {X.shape[0]} объектов, "
                         f"{X.shape[1]} признаков, {len(np.unique(y))} классов")

            return X, y

        except Exception as e:
            logger.error(f"Ошибка загрузки датасета {dataset_id}: {e}")
            return None, None
