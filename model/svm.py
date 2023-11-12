import pickle

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from joblib import dump, load

from const import svm_param_distributions, project_dir_path, cv, n_iters


class SVM:
    def __init__(self, model_name='support_vector_machine'):
        self._model = SVC()
        self._model_name = model_name

    def get_model(self):
        return self._model

    def save_model(self):
        dump(self._model, project_dir_path + '/model_parameters/' + self._model_name + '.joblib')

    def load_model(self):
        self._model = load(project_dir_path + '/model_parameters/' + self._model_name + '.joblib')

    def initialize_model_hyperparameters(self, **hyperparameters):
        self._model = SVC(**hyperparameters)

    def fit(self, X_train, y_train):
        self._model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._model.predict(X_test)

    def predict_proba(self, X_test):
        return self._model.predict_proba(X_test)

    def tune_hyperparameters_random_search(self, X, y, C, gamma, kernel, cv=5, n_jobs=-1, verbose=2, n_iter=100):
        param_distributions = {
            'C': C,
            'gamma': gamma,
            'kernel': kernel,
            'probability': [True]
        }

        random_search = RandomizedSearchCV(estimator=self._model, param_distributions=param_distributions, cv=cv,
                                           n_jobs=n_jobs, verbose=verbose, n_iter=n_iter)
        random_search.fit(X, y)
        return random_search.best_params_

    def tune(self, X, y):
        params = svm_param_distributions
        best_params = self.tune_hyperparameters_random_search(X, y, params['C'], params['gamma'], params['kernel'],
                                                              cv=cv, n_jobs=-1, verbose=2, n_iter=n_iters)
        with open(project_dir_path + '/model_hyperparameters/support_vector_machine.pkl', 'wb') as f:
            pickle.dump(best_params, f)
            f.close()
        self.initialize_model_hyperparameters(**best_params)
