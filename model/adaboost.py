import pickle

from sklearn.model_selection import RandomizedSearchCV
from joblib import dump, load
from const import project_dir_path, decision_tree_params, logistic_regression_params, ada_boost_base_params, cv, n_iters
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class AdaptiveBoostingClassifier:
    def __init__(self, model_name='adaptive_boost_using_decision_tree'):
        self._model_name = model_name
        self._base_model_name = model_name.split('using_')[-1]
        if self._base_model_name == 'decision_tree':
            self._base_model = DecisionTreeClassifier()
        elif self._base_model_name == 'logistic_regression':
            self._base_model = LogisticRegression()
        self._model = AdaBoostClassifier(estimator=self._base_model)

    def get_model(self):
        return self._model

    def save_model(self):
        dump(self._model, project_dir_path + '/model_parameters/' + self._model_name + '.joblib')

    def load_model(self):
        self._model = load(project_dir_path + '/model_parameters/' + self._model_name + '.joblib')

    def initialize_model_hyperparameters(self, **hyperparameters):
        use_decision_tree = 'base_estimator__max_depth' in hyperparameters.keys()
        base_estimator_params = {key.replace('base_estimator__', ''): value
                                 for key, value in hyperparameters.items() if key.startswith('base_estimator__')}
        if use_decision_tree:
            base_estimator = DecisionTreeClassifier(**base_estimator_params)
        else:
            base_estimator = LogisticRegression(**base_estimator_params)
        ada_params = {key: value for key, value in hyperparameters.items() if not key.startswith('base_estimator__')}
        self._model = AdaBoostClassifier(estimator=base_estimator, **ada_params)

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
        base_estimator_params = decision_tree_params if self._base_model_name == 'decision_tree'\
            else logistic_regression_params
        prefixed_base_params = {'base_estimator__' + key: value for key, value in base_estimator_params.items()}
        combined_params = {**ada_boost_base_params, **prefixed_base_params}
        best_params = self.tune_hyperparameters_random_search(X, y, combined_params, cv=cv,
                                                              n_jobs=-1, verbose=2, n_iter=n_iters)
        with open(project_dir_path + '/model_hyperparameters/' + self._model_name + '.pkl', 'wb') as f:
            pickle.dump(best_params, f)
            f.close()
        self.initialize_model_hyperparameters(**best_params)
