from const import data_dir_path, source_dir, individual_window_size, head_to_head_window_size, last_n_data

import pandas as pd
import os.path as osp

from data_loader.feature_generator import FeatureGenerator


def generate_features():
    match_results_file = osp.join(data_dir_path, source_dir, 'match_results.csv')
    match_results_df = pd.read_csv(match_results_file)
    feature_generator = FeatureGenerator(individual_window_size, head_to_head_window_size)
    features = feature_generator(match_results_df[-last_n_data:])
    features.to_csv(osp.join(data_dir_path, 'features_last_' + str(last_n_data) + '_windows_' + str(
        individual_window_size) + '_' + str(head_to_head_window_size) + '.csv'), index=False)
