from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
from tqdm import tqdm


class PCATransform:

    def __init__(self, n_components=5):
        self._n_components = n_components
        self._pca = PCA(n_components=self._n_components)
        self._scaler = StandardScaler()

    def fit_transform(self, X):
        X = self._scaler.fit_transform(X)
        return self._pca.fit_transform(X)

    def transform(self, X):
        X = self._scaler.transform(X)
        return self._pca.transform(X)

    def fit(self, X):
        X = self._scaler.fit_transform(X)
        self._pca.fit(X)

    def tune_n_components(self, X, n_components_min, n_components_max):
        components = np.arange(n_components_min, n_components_max)
        X = self._scaler.fit_transform(X)
        var_values = {}
        for component in tqdm(components):
            pca = PCA(n_components=component)
            pca.fit(X)
            var_values[component] = pca.explained_variance_ratio_.sum()
        return var_values
