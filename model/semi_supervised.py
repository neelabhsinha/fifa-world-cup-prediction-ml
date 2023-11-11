from sklearn.semi_supervised import SelfTrainingClassifier


class SemiSupervisedClassifier:
    def __init__(self, base_model, threshold=0.75):
        self._base_model = base_model
        self._self_training_model = SelfTrainingClassifier(base_model.get_model(), threshold=threshold, criterion='threshold')

    def get_model(self):
        return self._base_model

    def get_self_training_model(self):
        return self._self_training_model

    def fit(self, X_train, y_train):
        self._self_training_model.fit(X_train, y_train)

    def predict(self, X_test):
        return self._self_training_model.predict(X_test)

    def predict_proba(self, X_test):
        return self._self_training_model.predict_proba(X_test)
