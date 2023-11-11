import pickle

from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from const import decision_tree_params, cv, n_iters, project_dir_path


class DecisionTree:
    def __init__(self):
        self._model = DecisionTreeClassifier()

    def get_model(self):
        return self._model

    def initialize_model_hyperparameters(self, **hyperparameters):
        self._model = DecisionTreeClassifier(**hyperparameters)

    def fit(self, X_train, y_train):
        self._model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._model.predict(X_test)

    def predict_proba(self, X_test):
        return self._model.predict_proba(X_test)

    def tune_hyperparameters_random_search(self, X, y, max_depth, min_samples_split, min_samples_leaf, criterion, cv=5,
                                           n_jobs=-1, verbose=2, n_iter=100):
        param_grid = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'criterion': criterion
        }
        random_search = RandomizedSearchCV(estimator=self._model, param_distributions=param_grid, cv=cv, n_jobs=n_jobs,
                                           verbose=verbose, n_iter=n_iter)
        random_search.fit(X, y)
        return random_search.best_params_

    def tune(self, X, y):
        params = decision_tree_params
        best_params = self.tune_hyperparameters_random_search(X, y, params['max_depth'], params['min_samples_split'],
                                                              params['min_samples_leaf'], params['criterion'], cv=cv,
                                                              n_jobs=-1, verbose=2, n_iter=n_iters)
        with open(project_dir_path + '/model_hyperparameters/decision_tree.pkl', 'wb') as f:
            pickle.dump(best_params, f)
            f.close()
        self.initialize_model_hyperparameters(**best_params)

