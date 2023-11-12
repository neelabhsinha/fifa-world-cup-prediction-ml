import os.path

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, f1_score, \
    precision_score, recall_score, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from sklearn.tree import plot_tree

from const import project_dir_path

import matplotlib.pyplot as plt
import numpy as np


class ClassificationStatistics:

    def __init__(self, model, model_name, X_train, y_train, X_test, y_actual, y_predicted, y_predicted_proba,
                 averaging='weighted'):
        self._X_test = X_test
        self._y_actual = y_actual
        self._y_predicted = y_predicted
        self._y_predicted_proba = y_predicted_proba
        self._averaging = averaging
        self._model = model
        self._X_train = X_train
        self._y_train = y_train
        self._model_name = ' '.join(word.capitalize() for word in model_name.split('_'))

    def get_confusion_matrix(self):
        cm = confusion_matrix(self._y_actual, self._y_predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0.0, 1.0])
        disp.plot()
        if not os.path.exists(project_dir_path + '/results/' + self._model_name):
            os.mkdir(project_dir_path + '/results/' + self._model_name)
        plt.savefig(project_dir_path + '/results/' + self._model_name + '/confusion_matrix.png')
        plt.close()
        return cm

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

    def evaluate_model(self, extract_learning_curve=True):
        print('Results for Model: ', self._model_name)
        print('Confusion Matrix: \n', self.get_confusion_matrix())
        print('Classification Report: \n', self.get_classification_report())
        print('Accuracy Score: ', self.get_accuracy_score())
        print('ROC AUC Score: ', self.get_roc_auc_score())
        print('F1 Score: ', self.get_f1_score())
        print('Precision Score: ', self.get_precision_score())
        print('Recall Score: ', self.get_recall_score())
        if extract_learning_curve:
            self.save_learning_curve()
        self.save_decision_graph_for_decision_tree()

    def save_roc_curve(self, fpr, tpr, roc_auc):
        plt.title('Receiver Operating Characteristic for ' + self._model_name)
        plt.plot(tpr, fpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('False Positive Rate')
        plt.xlabel('True Positive Rate')
        if not os.path.exists(project_dir_path + '/results/' + self._model_name):
            os.mkdir(project_dir_path + '/results/' + self._model_name)
        plt.savefig(project_dir_path + '/results/' + self._model_name + '/roc_curve.png')
        plt.close()

    def save_learning_curve(self):
        print('Saving learning curve')
        model = self._model.get_model()
        train_sizes, train_scores, val_scores = learning_curve(
            model, self._X_train, self._y_train, cv=10, n_jobs=-1,
            train_sizes=np.linspace(.1, 1.0, 5),
            scoring='accuracy'
        )
        train_mean = np.mean(train_scores, axis=1) * 100
        train_std = np.std(train_scores, axis=1) * 100
        val_mean = np.mean(val_scores, axis=1) * 100
        val_std = np.std(val_scores, axis=1) * 100
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training accuracy')
        plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation accuracy')
        plt.title('Learning Curve for ' + self._model_name)
        plt.xlabel('Training examples')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='best')
        if not os.path.exists(project_dir_path + '/results/' + self._model_name):
            os.mkdir(project_dir_path + '/results/' + self._model_name)
        plt.savefig(project_dir_path + '/results/' + self._model_name + '/learning_curve.png')
        plt.close()

    def save_decision_graph_for_decision_tree(self):
        if 'decision' not in self._model_name.lower():
            return
        if not os.path.exists(project_dir_path + '/results/' + self._model_name):
            os.mkdir(project_dir_path + '/results/' + self._model_name)
        plt.figure(figsize=(150, 30))
        plot_tree(self._model.get_model(), filled=True, rounded=True, fontsize=13,
                  class_names=['0', '1'],
                  feature_names=self._X_test.columns.tolist())
        plt.savefig(project_dir_path + '/results/' + self._model_name + '/decision_tree.png')
        plt.close()
