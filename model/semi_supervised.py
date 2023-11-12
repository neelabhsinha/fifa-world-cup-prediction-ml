from sklearn.semi_supervised import SelfTrainingClassifier
from joblib import dump, load
from const import project_dir_path


class SemiSupervisedClassifier:
    def __init__(self, base_model, model_name, threshold=0.75):
        self._base_model = base_model
        self._self_training_model = SelfTrainingClassifier(base_model.get_model(), threshold=threshold, criterion='threshold')
        self._model_name = model_name

    def get_model(self):
        return self._base_model

    def save_model(self):
        dump(self._self_training_model, project_dir_path + '/model_parameters/' + self._model_name + '.joblib')

    def load_model(self):
        self._self_training_model = load(project_dir_path + '/model_parameters/' + self._model_name + '.joblib')

    def get_self_training_model(self):
        return self._self_training_model

    def fit(self, X_train, y_train):
        self._self_training_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._self_training_model.predict(X_test)

    def predict_proba(self, X_test):
        return self._self_training_model.predict_proba(X_test)
