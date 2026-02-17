import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
from pathlib import Path


class DimReductionPlotter:
    """Визуализация снижения размерности"""

    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None

    def plot_pca(self, df: pd.DataFrame, target_col: str = 'best_algorithm'):
        """PCA визуализация"""
        features = df.drop(['dataset_id', 'dataset_name', target_col],
                           axis=1, errors='ignore')
        features = features.select_dtypes(include=[np.number])
        features = features.fillna(features.mean())

        if features.shape[1] < 2:
            print("Недостаточно признаков для PCA")
            return

        # Масштабирование
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # Создание фигуры с двумя подграфиками
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 2D проекция
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                                  c=pd.factorize(df[target_col])[0],
                                  cmap='Set1', alpha=0.7, s=50)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        axes[0].set_title('PCA проекция')
        plt.colorbar(scatter, ax=axes[0], label=target_col)

        # Объясненная дисперсия
        axes[1].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                    pca.explained_variance_ratio_, alpha=0.7)
        axes[1].set_xlabel('Компонента')
        axes[1].set_ylabel('Доля объясненной дисперсии')
        axes[1].set_title('Объясненная дисперсия по компонентам')

        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / 'pca_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_tsne(self, df: pd.DataFrame, target_col: str = 'best_algorithm',
                  perplexities: list = [30, 50]):
        """t-SNE визуализация с разными perplexity"""
        features = df.drop(['dataset_id', 'dataset_name', target_col],
                           axis=1, errors='ignore')
        features = features.select_dtypes(include=[np.number])
        features = features.fillna(features.mean())

        if features.shape[1] < 2:
            print("Недостаточно признаков для t-SNE")
            return

        # Масштабирование
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        fig, axes = plt.subplots(1, len(perplexities), figsize=(6 * len(perplexities), 5))
        if len(perplexities) == 1:
            axes = [axes]

        for i, perp in enumerate(perplexities):
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
            X_tsne = tsne.fit_transform(X_scaled)

            scatter = axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1],
                                      c=pd.factorize(df[target_col])[0],
                                      cmap='Set1', alpha=0.7, s=50)
            axes[i].set_xlabel('t-SNE1')
            axes[i].set_ylabel('t-SNE2')
            axes[i].set_title(f't-SNE (perplexity={perp})')
            plt.colorbar(scatter, ax=axes[i], label=target_col)

        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / 'tsne_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()