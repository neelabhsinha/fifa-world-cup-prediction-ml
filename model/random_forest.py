from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class RandomForest:

    def __init__(self):
        self._model = RandomForestClassifier()

    def initialize_model_hyperparameters(self, **hyperparameters):
        self._model = RandomForestClassifier(**hyperparameters)

    def fit(self, X_train, y_train):
        self._model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._model.predict(X_test)

    def predict_proba(self, X_test):
        return self._model.predict_proba(X_test)

    def tune_hyperparameters_grid_search(self, X, y, n_estimators, min_sample_split, min_samples_leaf, max_depth, max_features,
                             bootstrap, cv=3, n_jobs=-1, verbose=2):
        param_grid = {
            'n_estimators': n_estimators,
            'min_samples_split': min_sample_split,
            'min_samples_leaf': min_samples_leaf,
            'max_depth': max_depth,
            'max_features': max_features,
            'bootstrap': bootstrap
        }
        grid_search = GridSearchCV(estimator=self._model, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
        grid_search.fit(X, y)
        return grid_search.best_params_

    def tune_hyperparameters_random_search(self, X, y, n_estimators, min_sample_split, min_samples_leaf, max_depth, max_features,
                                bootstrap, cv=5, n_jobs=-1, verbose=2, n_iter=100):
            param_grid = {
                'n_estimators': n_estimators,
                'min_samples_split': min_sample_split,
                'min_samples_leaf': min_samples_leaf,
                'max_depth': max_depth,
                'max_features': max_features,
                'bootstrap': bootstrap
            }
            random_search = RandomizedSearchCV(estimator=self._model, param_distributions=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose, n_iter=n_iter)
            random_search.fit(X, y)
            return random_search.best_params_

    def get_feature_importance(self, X):
        return self._model.feature_importances_

    def score(self, X_test, y_test):
        return self._model.score(X_test, y_test)

