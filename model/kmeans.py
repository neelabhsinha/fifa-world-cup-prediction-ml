from k_means_constrained import KMeansConstrained
import numpy as np

class Kmeans:
    def __init__(self, n_components=4, model_name= 'kmeans'):
        self._n_components = n_components
        self._model = KMeansConstrained(n_clusters=self._n_components ,size_min=8,size_max=8,random_state=0)
        self._model_name = model_name

    def fit(self, X_train):
        self._model.fit_predict(X_train)

    def get_clusters(self, features, all_countries):
        self._model.fit(features)

        sorted_cluster = np.argsort(self.model.cluster_centers_[:, -1]) #sort clusters according to rank

        for cluster in sorted_cluster:
            team_index = np.ravel(np.argwhere(self.model.labels_==cluster))
            random_team_index = np.random.permutation(team_index)
            team_cluster_index = np.concatenate((team_cluster_index, random_team_index), axis=0)

        clusters = np.array(all_countries)[team_cluster_index]
        return clusters.reshape(4,8)
