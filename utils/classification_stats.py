import os.path

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, f1_score, \
    precision_score, recall_score, roc_curve, auc
from const import project_dir_path

import matplotlib.pyplot as plt


class ClassificationStatistics:

    def __init__(self, model, X_test, y_actual, y_predicted, y_predicted_proba, averaging='weighted'):
        self._X_test = X_test
        self._y_actual = y_actual
        self._y_predicted = y_predicted
        self._y_predicted_proba = y_predicted_proba
        self._averaging = averaging
        self.__model = model

    def get_confusion_matrix(self):
        return confusion_matrix(self._y_actual, self._y_predicted)

    def get_classification_report(self):
        return classification_report(self._y_actual, self._y_predicted)

    def get_accuracy_score(self):
        return accuracy_score(self._y_actual, self._y_predicted)

    def get_roc_auc_score(self):
        fpr, tpr, thresholds = roc_curve(self._y_actual, self._y_predicted_proba[:, 0])
        auc_score = auc(fpr, tpr)
        self.save_roc_curve(fpr, tpr, auc_score)
        return roc_auc_score(self._y_actual, self._y_predicted_proba[:, 0])

    def get_f1_score(self):
        return f1_score(self._y_actual, self._y_predicted, average=self._averaging)

    def get_precision_score(self):
        return precision_score(self._y_actual, self._y_predicted, average=self._averaging)

    def get_recall_score(self):
        return recall_score(self._y_actual, self._y_predicted, average=self._averaging)

    def evaluate_model(self):
        print('Results for Model: ', self.__model)
        print('Confusion Matrix: \n', self.get_confusion_matrix())
        print('Classification Report: \n', self.get_classification_report())
        print('Accuracy Score: ', self.get_accuracy_score())
        print('ROC AUC Score: ', self.get_roc_auc_score())
        print('F1 Score: ', self.get_f1_score())
        print('Precision Score: ', self.get_precision_score())
        print('Recall Score: ', self.get_recall_score())

    def save_roc_curve(self, fpr, tpr, roc_auc):
        plt.title('Receiver Operating Characteristic for ' + self.__model)
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        if not os.path.exists(project_dir_path + '/results/' + self.__model):
            os.mkdir(project_dir_path + '/results/' + self.__model)
        plt.savefig(project_dir_path + '/results/' + self.__model + '/roc_curve.png')
