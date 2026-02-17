import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.meta_features.extractor import MetaFeatureExtractor
from src.meta_features.statistical import StatisticalFeatures
from src.meta_features.structural import StructuralFeatures


@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, random_state=42)
    return X, y


class TestMetaFeatureExtractor:
    """Тесты для MetaFeatureExtractor"""

    def test_extract(self, sample_data):
        X, y = sample_data
        extractor = MetaFeatureExtractor()
        features = extractor.extract(X, y)

        assert isinstance(features, dict)
        assert len(features) > 0
        assert 'n_instances' in features
        assert features['n_instances'] == 100


class TestStatisticalFeatures:
    """Тесты для StatisticalFeatures"""

    def test_extract(self, sample_data):
        X, y = sample_data
        extractor = StatisticalFeatures()
        features = extractor.extract(X)

        assert 'mean_mean' in features
        assert 'std_mean' in features


class TestStructuralFeatures:
    """Тесты для StructuralFeatures"""

    def test_extract(self, sample_data):
        X, y = sample_data
        extractor = StructuralFeatures()
        features = extractor.extract(X, y)

        assert 'class_entropy' in features
        assert features['class_entropy'] > 0