import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class MetaVisualizer:
    """Главный класс для визуализации"""

    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        sns.set_style('whitegrid')

    def plot_correlation_matrix(self, df: pd.DataFrame, figsize=(14, 12)):
        df_numeric = df.select_dtypes(include=[np.number])

        if df_numeric.shape[1] < 2:
            print("Недостаточно признаков для корреляционной матрицы")
            return

        corr = df_numeric.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=figsize)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, square=True)
        plt.title('Корреляционная матрица мета-признаков')
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_pca_projection(self, df: pd.DataFrame, target_col='best_algorithm'):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        features = df.drop(['dataset_id', 'dataset_name', target_col],
                           axis=1, errors='ignore').select_dtypes(include=[np.number])
        features = features.fillna(features.mean())

        if features.shape[1] < 2:
            print("Недостаточно признаков для PCA")
            return

        X_scaled = StandardScaler().fit_transform(features)
        X_pca = PCA(n_components=2).fit_transform(X_scaled)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                              c=pd.factorize(df[target_col])[0],
                              cmap='Set1', alpha=0.7, s=50)
        plt.colorbar(scatter, label=target_col)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA проекция мета-набора')

        if self.save_dir:
            plt.savefig(self.save_dir / 'pca_projection.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_tsne_projection(self, df: pd.DataFrame, target_col='best_algorithm', perplexity=30):
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        features = df.drop(['dataset_id', 'dataset_name', target_col],
                           axis=1, errors='ignore').select_dtypes(include=[np.number])
        features = features.fillna(features.mean())

        if features.shape[1] < 2:
            print("Недостаточно признаков для t-SNE")
            return

        X_scaled = StandardScaler().fit_transform(features)
        X_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(X_scaled)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                              c=pd.factorize(df[target_col])[0],
                              cmap='Set1', alpha=0.7, s=50)
        plt.colorbar(scatter, label=target_col)
        plt.xlabel('t-SNE1')
        plt.ylabel('t-SNE2')
        plt.title(f't-SNE проекция (perplexity={perplexity})')

        if self.save_dir:
            plt.savefig(self.save_dir / 'tsne_projection.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_algorithm_performance(self, results_df: pd.DataFrame):
        score_cols = [c for c in results_df.columns if c.startswith('score_')]

        if not score_cols:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Box plot
        data = []
        for col in score_cols:
            for score in results_df[col].dropna():
                data.append({'Algorithm': col.replace('score_', ''), 'Score': score})

        perf_df = pd.DataFrame(data)
        sns.boxplot(data=perf_df, x='Algorithm', y='Score', ax=axes[0])
        axes[0].set_title('Распределение производительности алгоритмов')
        axes[0].tick_params(axis='x', rotation=45)

        # Best algorithm distribution
        if 'best_algorithm' in results_df.columns:
            counts = results_df['best_algorithm'].value_counts()
            counts.plot(kind='bar', ax=axes[1], color='coral', alpha=0.7)
            axes[1].set_title('Распределение лучших алгоритмов')
            axes[1].set_xlabel('Алгоритм')
            axes[1].set_ylabel('Количество датасетов')
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / 'algorithm_performance.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_noise_experiment(self, changes_df: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Изменения признаков
        changes_df['abs_change'] = changes_df['change'].abs()
        top_changes = changes_df.nlargest(10, 'abs_change')

        axes[0].barh(range(len(top_changes)), top_changes['abs_change'], color='coral')
        axes[0].set_yticks(range(len(top_changes)))
        axes[0].set_yticklabels(top_changes['feature'])
        axes[0].set_xlabel('Абсолютное изменение')
        axes[0].set_title('Топ-10 наиболее изменившихся признаков')

        # Распределение изменений
        axes[1].hist(changes_df['change'], bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='red', linestyle='--')
        axes[1].set_xlabel('Изменение значения')
        axes[1].set_ylabel('Частота')
        axes[1].set_title('Распределение изменений мета-признаков')

        plt.suptitle('Эксперимент с шумом')
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(self.save_dir / 'noise_experiment.png', dpi=150, bbox_inches='tight')
        plt.show()