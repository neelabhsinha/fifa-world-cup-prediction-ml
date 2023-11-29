import numpy as np

from const import project_dir_path
from model.decision_tree import DecisionTree
from model.gradient_boost import GradientBoost
from model.knn import KNearestNeighbours
from model.logistic_regression_class import LogisticRegressionClass
from model.naive_bayes import NaiveBayesClassifier
from model.random_forest import RandomForest
from model.svm import SVM


def _load_classifier(classifier_name):
    model = None
    if classifier_name == 'support_vector_machine':
        model = SVM()
    elif classifier_name == 'decision_tree':
        model = DecisionTree()
    elif classifier_name == 'logistic_regression':
        model = LogisticRegressionClass()
    elif classifier_name == 'k_nearest_neighbours':
        model = KNearestNeighbours()
    elif classifier_name == 'naive_bayes_classifier':
        model = NaiveBayesClassifier()
    elif classifier_name == 'random_forest':
        model = RandomForest()
    elif classifier_name == 'gradient_boost':
        model = GradientBoost()
    model.load_model()
    return model


class EnsembleClassifier:
    def __init__(self):
        self._classifiers = []
        self.load_model()

    def get_model(self):
        return self
    def save_model(self):
        print('Ensemble classifier does not support saving model')

    def load_model(self):
        self._classifiers.append(_load_classifier('support_vector_machine'))
        self._classifiers.append(_load_classifier('decision_tree'))
        self._classifiers.append(_load_classifier('logistic_regression'))
        self._classifiers.append(_load_classifier('k_nearest_neighbours'))
        self._classifiers.append(_load_classifier('naive_bayes_classifier'))

    def initialize_model_hyperparameters(self, **hyperparameters):
        print('Ensemble classifier does not support hyperparameter tuning')

    def fit(self, X_train, y_train):
        for clf in self._classifiers:
            clf.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = np.asarray([clf.predict(X_test) for clf in self._classifiers]).astype(int)
        majority_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
        return majority_vote

    def predict_proba(self, X_test):
        probas = np.asarray([clf.predict_proba(X_test) for clf in self._classifiers])
        avg_proba = np.mean(probas, axis=0)
        return avg_proba
