import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from const import random_forest_params, project_dir_path, n_iters, cv


class RandomForest:

    def __init__(self):
        self._model = RandomForestClassifier()

    def get_model(self):
        return self._model

    def initialize_model_hyperparameters(self, **hyperparameters):
        self._model = RandomForestClassifier(**hyperparameters)

    def fit(self, X_train, y_train):
        self._model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._model.predict(X_test)

    def predict_proba(self, X_test):
        return self._model.predict_proba(X_test)

    def tune_hyperparameters_grid_search(self, X, y, n_estimators, min_samples_split, min_samples_leaf, max_depth, max_features,
                             bootstrap, cv=3, n_jobs=-1, verbose=2):
        param_grid = {
            'n_estimators': n_estimators,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_depth': max_depth,
            'max_features': max_features,
            'bootstrap': bootstrap
        }
        grid_search = GridSearchCV(estimator=self._model, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose)
        grid_search.fit(X, y)
        return grid_search.best_params_

    def tune_hyperparameters_random_search(self, X, y, n_estimators, min_samples_split, min_samples_leaf, max_depth, max_features,
                                bootstrap, cv=5, n_jobs=-1, verbose=2, n_iter=100):
            param_grid = {
                'n_estimators': n_estimators,
                'min_samples_split': min_samples_split,
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

    def tune(self,X, y):
        params = random_forest_params
        best_params_range = self.tune_hyperparameters_random_search(X, y, n_estimators=params['n_estimators'], min_samples_split=params['min_samples_split'],
                                                                    min_samples_leaf=params['min_samples_leaf'], max_depth=params['max_depth'],
                                                                    max_features=params['max_features'], bootstrap=params['bootstrap'], cv=cv, n_iter=n_iters)
        print('Best hyperparameters for Random Forest using random search: ', best_params_range)
        # best_params_range['n_estimators'] = np.linspace(best_params_range['n_estimators'] - 100,best_params_range['n_estimators'] + 100, 10)
        # best_params_range['min_samples_split'] = np.linspace(best_params_range['min_samples_split'] - 2, best_params_range['min_samples_split'] + 2)
        # best_params_range['min_samples_leaf'] = np.linspace(best_params_range['min_samples_leaf'] - 2, best_params_range['min_samples_leaf'] + 2)
        # best_params_range['max_depth'] = np.linspace(best_params_range['max_depth'] - 10, best_params_range['max_depth'] + 10)
        # best_params_range['max_features'] = [best_params_range['max_features']]
        # best_params_range['bootstrap'] = [best_params_range['bootstrap']]
        # best_params = self.tune_hyperparameters_grid_search(X, y, best_params_range['n_estimators'], best_params_range['min_samples_split'],
        #                                                     best_params_range['min_samples_leaf'], best_params_range['max_depth'],
        #                                                     best_params_range['max_features'], best_params_range['bootstrap'], cv=cv, n_jobs=-1, verbose=2)
        # print('Best hyperparameters for Random Forest using grid search: ', best_params)
        best_params = best_params_range
        with open(project_dir_path + '/model_hyperparameters/random_forest.pkl', 'wb') as f:
            pickle.dump(best_params, f)
            f.close()
        self.initialize_model_hyperparameters(**best_params)


