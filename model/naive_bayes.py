import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump, load
from const import project_dir_path, naive_bayes_params, cv, n_iters


class NaiveBayesClassifier:
    def __init__(self, model_name='naive_bayes_classifier'):
        self._model = GaussianNB()
        self._model_name = model_name

    def get_model(self):
        return self._model

    def save_model(self):
        dump(self._model, project_dir_path + '/model_parameters/' + self._model_name + '.joblib')

    def load_model(self):
        self._model = load(project_dir_path + '/model_parameters/' + self._model_name + '.joblib')

    def initialize_model_hyperparameters(self, **hyperparameters):
        self._model = GaussianNB(**hyperparameters)

    def fit(self, X_train, y_train):
        self._model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._model.predict(X_test)

    def predict_proba(self, X_test):
        return self._model.predict_proba(X_test)

    def tune_hyperparameters_random_search(self, X, y, param_grid, cv=5, n_jobs=-1, verbose=2, n_iter=100):
        random_search = RandomizedSearchCV(estimator=self._model, param_distributions=param_grid, cv=cv, n_jobs=n_jobs,
                                           verbose=verbose, n_iter=n_iter)
        random_search.fit(X, y)
        return random_search.best_params_

    def tune(self, X, y):
        params = naive_bayes_params
        best_params = self.tune_hyperparameters_random_search(X, y, params, cv=cv,
                                                              n_jobs=-1, verbose=2, n_iter=n_iters)
        with open(project_dir_path + '/model_hyperparameters/' + self._model_name + '.pkl', 'wb') as f:
            pickle.dump(best_params, f)
            f.close()
        self.initialize_model_hyperparameters(**best_params)
