from typing import Dict, List, Optional


class FeatureGroups:
    """Группы мета-признаков"""

    GENERAL = ['n_features', 'n_instances', 'n_classes', 'dimensionality']
    STATISTICAL = ['mean', 'sd', 'skewness', 'kurtosis', 'correlation_mean']
    INFORMATION = ['class_entropy', 'mutual_info_mean', 'noise_signal_ratio']
    LANDMARKING = ['decision_tree_accuracy', 'naive_bayes_accuracy', 'one_nn_accuracy']

    def __init__(self):
        self.groups = {
            'general': self.GENERAL,
            'statistical': self.STATISTICAL,
            'information': self.INFORMATION,
            'landmarking': self.LANDMARKING
        }

    def get_group(self, name: str) -> List[str]:
        return self.groups.get(name, [])

    def get_all_features(self) -> List[str]:
        all_features = []
        for features in self.groups.values():
            all_features.extend(features)
        return list(set(all_features))