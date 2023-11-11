from model.gradient_boost import GradientBoost
from model.random_forest import RandomForest
from model.pca import PCATransform
from const import project_dir_path, data_dir_path, individual_window_size, head_to_head_window_size, last_n_data, \
    pca_n_components
from data_loader.data_split import split_feature_and_labels, get_train_test_split
from model.semi_supervised import SemiSupervisedClassifier
from model.svm import SVM
from utils.classification_stats import ClassificationStatistics
from data_loader.feature_selector import FeatureSelector

import pandas as pd
import os.path as osp

from utils.preprocess import generate_features, generate_artificial_matches


def train(model_name, do_pca=False, tune=False, select_features=True, semi_supervised=False):
    model = None
    X, y = load_data(artificial=False)
    print('Data loaded')
    if model_name == 'random_forest':
        model = RandomForest()
    elif model_name == 'gradient_boost':
        model = GradientBoost()
    elif model_name == 'svm':
        model = SVM()
    if select_features:
        print('Selecting features')
        feature_selector = FeatureSelector(model, X, y)
        selected_features = feature_selector.select_features()
    if do_pca:
        print('Performing PCA')
        pca = PCATransform(n_components=pca_n_components)
        pca.fit(X)
        pca_X = pca.transform(X)
    X = X[selected_features] if select_features else X
    x_train, x_test, y_train, y_test = get_train_test_split(pca_X if do_pca else X, y)
    if tune:
        model.tune(x_train, y_train)
    else:
        hyperparameters = load_model(model_name)
        if hyperparameters is not None:
            model.initialize_model_hyperparameters(**hyperparameters)
        else:
            model.tune(x_train, y_train)
    train_and_evaluate(model_name, model, x_train, y_train, x_test, y_test)
    if semi_supervised:
        model_name = model_name + '_semi_supervised'
        model = SemiSupervisedClassifier(base_model=model, threshold=0.75)
        artificial_X, artificial_y = load_data(artificial=True)
        x_train = pd.concat([x_train, artificial_X], ignore_index=True)
        artificial_y.loc[:] = -1
        y_train = pd.concat([y_train, artificial_y], ignore_index=True)
        train_and_evaluate(model_name, model, x_train, y_train, x_test, y_test, semi_supervised=True)


def train_and_evaluate(model_name, model, x_train, y_train, x_test, y_test, semi_supervised=False):
    print('Training model: ' + model_name)
    model.fit(x_train, y_train)
    print('Model trained')
    print('Evaluating Model')
    y_hat_test = model.predict(x_test)
    y_hat_test_proba = model.predict_proba(x_test)
    if semi_supervised:
        y_train = model.get_self_training_model().transduction_
        # model = model.get_model()
        # x_train = x_train[y_train != -1]
        # y_train = y_train[y_train != -1]
        # train_and_evaluate(model_name, model, x_train, y_train, x_test, y_test, semi_supervised=False)
    evaluator = ClassificationStatistics(model, model_name, x_train, y_train, x_test, y_test, y_hat_test,
                                         y_hat_test_proba)
    evaluator.evaluate_model(extract_learning_curve=True)


def load_model(model_name):
    try:
        hyperparameters = pd.read_pickle(project_dir_path + '/model_hyperparameters/' + model_name + '.pkl')
        return hyperparameters
    except FileNotFoundError:
        print('Hyperparameters for the model not found. Will need to tune the model.')
    return None


def load_data(artificial):
    print('Loading data')
    prefix = 'artificial_features_last_' if artificial else 'features_last_'
    try:
        features = pd.read_csv(osp.join(data_dir_path, prefix + str(last_n_data) + '_windows_' + str(
            individual_window_size) + '_' + str(head_to_head_window_size) + '.csv'))
    except FileNotFoundError:
        print('Features file not found.')
        if not artificial:
            generate_features()
        else:
            generate_artificial_matches()
        features = pd.read_csv(osp.join(data_dir_path, prefix + str(last_n_data) + '_windows_' + str(
            individual_window_size) + '_' + str(head_to_head_window_size) + '.csv'))
    X, y = split_feature_and_labels(features)
    y = y.iloc[:, :1].squeeze()
    return X, y
