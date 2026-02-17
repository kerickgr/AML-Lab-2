import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.algorithms.base_learners import BaseLearners
from src.algorithms.evaluation import AlgorithmEvaluator


@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    return X, y


class TestBaseLearners:
    """Тесты для BaseLearners"""

    def test_initialization(self):
        learners = BaseLearners(['LogisticRegression', 'RandomForest'])
        assert len(learners.get_all_algorithms()) == 2


class TestAlgorithmEvaluator:
    """Тесты для AlgorithmEvaluator"""

    def test_evaluate_all(self, sample_data):
        X, y = sample_data
        learners = BaseLearners(['LogisticRegression', 'RandomForest'])
        evaluator = AlgorithmEvaluator(learners.get_all_algorithms())

        scores = evaluator.evaluate_all(X, y)

        assert isinstance(scores, dict)
        assert len(scores) == 2
        assert all(0 <= v <= 1 for v in scores.values())