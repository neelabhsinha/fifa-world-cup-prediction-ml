from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, f1_score, \
    precision_score, recall_score


class ClassificationStatistics:

    def __init__(self, X_test, y_actual, y_predicted, y_predicted_proba, averaging = 'weighted'):
        self._X_test = X_test
        self._y_actual = y_actual
        self._y_predicted = y_predicted
        self._y_predicted_proba = y_predicted_proba
        self._averaging = averaging

    def get_confusion_matrix(self):
        return confusion_matrix(self._y_actual, self._y_predicted)

    def get_classification_report(self):
        return classification_report(self._y_actual, self._y_predicted)

    def get_accuracy_score(self):
        return accuracy_score(self._y_actual, self._y_predicted)

    def get_roc_auc_score(self):
        return roc_auc_score(self._y_actual, self._y_predicted_proba)

    def get_f1_score(self):
        return f1_score(self._y_actual, self._y_predicted, average=self._averaging)

    def get_precision_score(self):
        return precision_score(self._y_actual, self._y_predicted, average=self._averaging)

    def get_recall_score(self):
        return recall_score(self._y_actual, self._y_predicted, average=self._averaging)

