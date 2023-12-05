from k_means_constrained import KMeansConstrained
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from model.betaCV import betacv

class Kmeans:
    def __init__(self, n_components=4, model_name= 'kmeans'):
        self._n_components = n_components
        self._model = KMeansConstrained(n_clusters=self._n_components ,size_min=8,size_max=8,random_state=0)
        self._model_name = model_name

    def fit(self, X_train):
        self._model.fit_predict(X_train)

    def get_clusters(self, features, all_countries):
        self._model.fit(features)

        sorted_cluster = np.argsort(self._model.cluster_centers_[:, -1]) #sort clusters according to rank
        team_cluster_index = np.empty(0,dtype=int)
        
        for cluster in sorted_cluster:
            team_index = np.ravel(np.argwhere(self._model.labels_==cluster))
            random_team_index = np.random.permutation(team_index)
            team_cluster_index = np.concatenate((team_cluster_index, random_team_index), axis=0)

        clusters = np.array(all_countries)[team_cluster_index]
        print(f"Silhouette Score - {self.silhouette_average(features)}")
        print(f"Davies-Bouldin Index - {self.db_index_score(features)}")
        print(f"Beta CV measure - {self.betaCV_score(features)}")
        return clusters.reshape(4,8)

    def silhouette_average(self, X):
        labels = self._model.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        return silhouette_avg
        
    def db_index_score(self, X):
        db_index = davies_bouldin_score(X, self._model.labels_)
        return db_index
    
    # from https://github.com/hayashikan/betacv/blob/master/betacv.py
    def betaCV_score(self,X):
        return betacv(X,self._model.labels_,metric='euclidean')
