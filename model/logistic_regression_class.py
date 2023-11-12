from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class LogisticRegressionClass:

    def __init__(self):
        self._model = LogisticRegression()

    def initialize_model_hyperparameters(self, **hyperparameters):
        self._model = LogisticRegression(**hyperparameters)

    def fit(self, X_train, y_train):
        self._model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._model.predict(X_test)

    def predict_proba(self, X_test):
        return self._model.predict_proba(X_test)

    def tune_hyperparameters_grid_search(self, X, y, solver, penalty, c_values, cv=3, n_jobs=-1, verbose=2):
        C = c_values
        param_grid = {
            'solver': solver,
            'penalty': penalty,
            'C': C
        }
        grid_search = GridSearchCV(estimator=self._model, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
        grid_search.fit(X, y)
        return grid_search.best_params_

    def tune_hyperparameters_random_search(self, X, y, solver, penalty, c_values,
                                cv=5, n_jobs=-1, verbose=2, n_iter=100):
        C = c_values    
        param_grid = {
            'solver': solver,
            'penalty': penalty,
            'C': C
        }
        random_search = RandomizedSearchCV(estimator=self._model, param_distributions=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose, n_iter=n_iter,error_score='raise')
        random_search.fit(X, y)
        return random_search.best_params_

    def get_feature_importance(self, X):
        return self._model.feature_importances_

    def score(self, X_test, y_test):
        return self._model.score(X_test, y_test)

