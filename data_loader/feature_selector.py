from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score
from data_loader.data_split import split_feature_and_labels, get_train_test_split


class FeatureSelector:
    def __init__(self, model, x_train, y_train):
        self._model = model
        self._x_train, self._x_cv, self._y_train, self._y_cv = get_train_test_split(x_train, y_train)
        self._feature_names = self._x_train.columns

    def select_features(self):
        selected_features = []
        best_accuracy = 0.0
        for feature in self._feature_names:
            best_feature = None
            for i in self._feature_names:
                if i not in selected_features:
                    trial_features = selected_features + [i]
                    self._model.fit(self._x_train[trial_features], self._y_train)
                    y_pred = self._model.predict(self._x_cv[trial_features])
                    accuracy = accuracy_score(self._y_cv, y_pred)
                    if accuracy > best_accuracy - 0.01:
                        best_feature = i
                        best_accuracy = accuracy
            if best_feature is not None:
                selected_features.append(best_feature)
            else:
                break
        return selected_features
