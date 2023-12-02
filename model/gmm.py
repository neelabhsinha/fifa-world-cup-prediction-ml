from sklearn.mixture import GaussianMixture
import numpy as np
from joblib import dump, load
from const import project_dir_path
from sklearn.metrics import silhouette_score


class GaussianMixtureModel:

    def __init__(self, n_components=4, model_name= 'gmm'):
        self._n_components = n_components
        self._model = GaussianMixture(n_components=self._n_components, random_state=0)
        self._model_name = model_name

    def fit(self, X_train):
        self._model.fit(X_train)

    def predict(self, X_test):
        return self._model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self._model.predict_proba(X_test)
    
    def save_model(self):
        dump(self._model, project_dir_path + '/model_parameters/' + self._model_name + '.joblib')

    def load_model(self):
        self._model = load(project_dir_path + '/model_parameters/' + self._model_name + '.joblib')

    def get_clusters(self, features, all_countries):
        self._model.fit(features)
        group_probs= self._model.predict_proba(features)
        map= {}
        clusters= []
        for i in range(4):
            sorted_indices = np.argsort(group_probs[:,i])[::-1]
            in_cluster= []
            j=0
            while len(in_cluster)!=8:
                idx= sorted_indices[j]
                if map.get(idx,-1)<0:
                    in_cluster.append(idx)
                    map[idx]=1
                j+=1
            clusters.append(np.array(all_countries)[in_cluster])
        print(f"Silhouette Score- {self.silhouette_average(features)}")
        return np.array(clusters)

    def silhouette_average(self, X):
        labels = self.predict(X)
        silhouette_avg = silhouette_score(X, labels)
        return silhouette_avg

