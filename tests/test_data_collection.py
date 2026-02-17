import pytest
import pandas as pd
import numpy as np
from src.data.collector import OpenMLCollector
from src.data.filters import DatasetFilter


class TestOpenMLCollector:
    """Тесты для OpenMLCollector"""

    def test_initialization(self):
        collector = OpenMLCollector(cache_dir='test_cache')
        assert collector.cache_dir.exists()

    def test_fetch_datasets_list(self):
        collector = OpenMLCollector()
        df = collector.fetch_datasets_list(limit=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 10


class TestDatasetFilter:
    """Тесты для DatasetFilter"""

    def test_filter_by_size(self):
        df = pd.DataFrame({
            'NumberOfInstances': [50, 100, 200, 500],
            'NumberOfFeatures': [5, 10, 15, 20]
        })

        filtered = DatasetFilter.filter_by_size(df, min_instances=100, max_instances=300)
        assert len(filtered) == 2
        assert filtered['NumberOfInstances'].tolist() == [100, 200]