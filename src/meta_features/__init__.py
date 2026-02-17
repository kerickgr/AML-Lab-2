"""
Модуль извлечения мета-признаков
"""
# Закомментируйте или удалите старый импорт
# from src.meta_features.extractor import MetaFeatureExtractor

# Добавьте новый импорт
from src.meta_features.fixed_extractor import FixedMetaFeatureExtractor as MetaFeatureExtractor
from src.meta_features.groups import FeatureGroups

__all__ = [
    "MetaFeatureExtractor",
    "FeatureGroups"
]
