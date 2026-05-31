"""
habitat_clustering.py
Radiomic habitat clustering pipeline for GBM tumour microenvironment analysis.
Author: Sylvester KT (@SylvesterKT)
"""

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

logger = logging.getLogger(__name__)


class HabitatClusterer:
    """
    Unsupervised radiomic habitat clustering for intra-tumoral heterogeneity analysis.

    Clusters patients into distinct radiomic habitats using K-means or
    hierarchical clustering on normalised radiomic feature matrices.

    Args:
        n_clusters:   Number of habitat clusters (default: 4)
        method:       Clustering algorithm: 'kmeans' or 'hierarchical'
        random_state: Random seed for reproducibility
        scale:        Whether to standardise features before clustering
    """

    def __init__(
        self,
        n_clusters: int = 4,
        method: str = "kmeans",
        random_state: int = 42,
        scale: bool = True,
    ):
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.scale = scale
        self.scaler = StandardScaler() if scale else None
        self.model = None
        self.labels_ = None
        self.feature_matrix_ = None

    def _build_model(self):
        if self.method == "kmeans":
            return KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=20,
                max_iter=500,
            )
        elif self.method == "hierarchical":
            return AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage="ward",
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'kmeans' or 'hierarchical'.")

    def fit_predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Fit the clustering model and return habitat labels.

        Args:
            features: pd.DataFrame of shape (n_patients, n_features)

        Returns:
            np.ndarray of cluster labels (0-indexed)
        """
        X = features.values.copy()

        # Handle missing values
        nan_mask = np.isnan(X)
        if nan_mask.any():
            col_means = np.nanmean(X, axis=0)
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
            logger.warning(f"Imputed {nan_mask.sum()} missing values with column means")

        if self.scale:
            X = self.scaler.fit_transform(X)

        self.feature_matrix_ = X
        self.model = self._build_model()
        self.labels_ = self.model.fit_predict(X)

        # Log cluster quality metrics
        sil = silhouette_score(X, self.labels_)
        ch = calinski_harabasz_score(X, self.labels_)
        logger.info(f"Clustering complete | Silhouette: {sil:.3f} | Calinski-Harabasz: {ch:.1f}")

        return self.labels_

    def optimal_k(
        self, features: pd.DataFrame, k_range: range = range(2, 10)
    ) -> Tuple[list, list]:
        """
        Evaluate silhouette scores across a range of k values.

        Args:
            features: Feature matrix
            k_range:  Range of k values to evaluate

        Returns:
            Tuple of (k_values, silhouette_scores)
        """
        X = features.values.copy()
        if self.scale:
            X = StandardScaler().fit_transform(X)

        scores = []
        for k in k_range:
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(X)
            scores.append(silhouette_score(X, labels))
            logger.info(f"k={k}: silhouette={scores[-1]:.3f}")

        return list(k_range), scores

    def plot_pca(
        self,
        features: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        PCA scatter plot coloured by habitat cluster.

        Args:
            features:  Feature matrix
            labels:    Cluster labels (uses self.labels_ if None)
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        X = self.feature_matrix_ if self.feature_matrix_ is not None else features.values
        labels = labels if labels is not None else self.labels_

        pca = PCA(n_components=2, random_state=self.random_state)
        X_2d = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=labels,
            cmap="tab10",
            s=60,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.3,
        )
        plt.colorbar(scatter, ax=ax, label="Habitat Cluster")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
        ax.set_title("GBM Radiomic Habitat Clusters (PCA)")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"PCA plot saved to {save_path}")

        return fig

    def get_habitat_summary(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-habitat mean feature profiles.

        Returns:
            pd.DataFrame with habitat as index and mean feature values as columns
        """
        df = features.copy()
        df["habitat"] = self.labels_
        summary = df.groupby("habitat").mean()
        summary.index = [f"Habitat-{i+1}" for i in summary.index]
        return summary
