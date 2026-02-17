import pandas as pd
from typing import Callable, Dict, List, Optional


class DatasetFilter:
    """Фильтрация датасетов"""

    @staticmethod
    def filter_by_size(df: pd.DataFrame, min_instances=100, max_instances=10000) -> pd.DataFrame:
        return df[
            (df['NumberOfInstances'] >= min_instances) &
            (df['NumberOfInstances'] <= max_instances)
            ]

    @staticmethod
    def filter_by_features(df: pd.DataFrame, min_features=3, max_features=500) -> pd.DataFrame:
        return df[
            (df['NumberOfFeatures'] >= min_features) &
            (df['NumberOfFeatures'] <= max_features)
            ]

    @staticmethod
    def filter_by_classes(df: pd.DataFrame, min_classes=2, max_classes=50) -> pd.DataFrame:
        return df[
            (df['NumberOfClasses'] >= min_classes) &
            (df['NumberOfClasses'] <= max_classes)
            ]