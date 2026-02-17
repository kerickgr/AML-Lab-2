"""
Тестовый скрипт для проверки первых 10 датасетов
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import logging
from src.data.collector import OpenMLCollector
from src.meta_features.extractor import MetaFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_first_10():
    """Тестирование первых 10 датасетов"""

    # Загружаем список датасетов
    datasets_df = pd.read_csv('data/raw/dataset_list.csv')
    first_10 = datasets_df.head(10)

    collector = OpenMLCollector(use_cache=True)
    extractor = MetaFeatureExtractor(use_cache=True)

    results = []

    for idx, row in first_10.iterrows():
        dataset_id = row['did']
        logger.info(f"\nТестирование датасета {dataset_id}: {row['name']}")

        try:
            X, y = collector.download_dataset(dataset_id)

            if X is None or y is None:
                logger.error(f"  ❌ Не удалось загрузить датасет")
                continue

            logger.info(f"  ✅ Загружен: {X.shape[0]} объектов, {X.shape[1]} признаков")
            logger.info(f"  ✅ Классов: {len(np.unique(y))}")

            features = extractor.extract(X, y, dataset_id=str(dataset_id))

            if features:
                logger.info(f"  ✅ Извлечено {len(features)} мета-признаков")
                results.append({
                    'dataset_id': dataset_id,
                    'dataset_name': row['name'],
                    **features
                })
            else:
                logger.error(f"  ❌ Не удалось извлечь мета-признаки")

        except Exception as e:
            logger.error(f"  ❌ Ошибка: {e}")

    # Сохраняем результаты
    if results:
        df = pd.DataFrame(results)
        df.to_csv('data/processed/test_results.csv', index=False)
        logger.info(f"\n✅ Сохранено {len(df)} результатов")
    else:
        logger.error("\n❌ Нет результатов")


if __name__ == "__main__":
    test_first_10()
