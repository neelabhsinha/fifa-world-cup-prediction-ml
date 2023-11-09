from model.gradient_boost import GradientBoost
from model.random_forest import RandomForest
from model.pca import PCATransform
from const import project_dir_path, data_dir_path, individual_window_size, head_to_head_window_size, last_n_data, \
    pca_n_components
from data_loader.data_split import split_feature_and_labels, get_train_test_split
from utils.classification_stats import ClassificationStatistics
from data_loader.feature_selector import FeatureSelector

import pandas as pd
import os.path as osp

from utils.preprocess import generate_features


def train(model_name, do_pca=False, tune=False, select_features=True):
    model = None
    print('Loading data')
    try:
        features = pd.read_csv(osp.join(data_dir_path, 'features_last_' + str(last_n_data) + '_windows_' + str(
            individual_window_size) + '_' + str(head_to_head_window_size) + '.csv'))
    except FileNotFoundError:
        print('Features file not found. Please generate the features first.')
        generate_features()
        features = pd.read_csv(osp.join(data_dir_path, 'features_last_' + str(last_n_data) + '_windows_' + str(
            individual_window_size) + '_' + str(head_to_head_window_size) + '.csv'))
    X_train, y_train = split_feature_and_labels(features)
    y_train = y_train.iloc[:, :1].squeeze()
    print('Data loaded')
    if model_name == 'random_forest':
        model = RandomForest()
    elif model_name == 'gradient_boost':
        model = GradientBoost()
    if select_features:
        print('Selecting features')
        feature_selector = FeatureSelector(model, X_train, y_train)
        selected_features = feature_selector.select_features()
    if do_pca:
        print('Performing PCA')
        pca = PCATransform(n_components=pca_n_components)
        pca.fit(X_train)
        pca_X_train = pca.transform(X_train)
    X_train = X_train[selected_features] if select_features else X_train
    x_train, x_test, y_train, y_test = get_train_test_split(pca_X_train if do_pca else X_train, y_train)
    if tune:
        model.tune(x_train, y_train)
    else:
        hyperparameters = load_model(model_name)
        if hyperparameters is not None:
            model.initialize_model_hyperparameters(**hyperparameters)
        else:
            model.tune(x_train, y_train)
    print('Training model')
    model.fit(x_train, y_train)
    print('Model trained')
    print('Evaluating Model')
    y_hat_test = model.predict(x_test)
    y_hat_test_proba = model.predict_proba(x_test)
    evaluator = ClassificationStatistics(model_name, x_test, y_test, y_hat_test, y_hat_test_proba)
    evaluator.evaluate_model()


def load_model(model_name):
    try:
        hyperparameters = pd.read_pickle(project_dir_path + '/model_hyperparameters/' + model_name + '.pkl')
        return hyperparameters
    except FileNotFoundError:
        print('Hyperparameters for the model not found. Will need to tune the model.')
    return None
