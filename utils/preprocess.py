from const import data_dir_path, source_dir, individual_window_size, head_to_head_window_size, last_n_data

import pandas as pd
import os.path as osp

from feature.feature_generator import FeatureGenerator
from feature.artificial_data_generator import ArtificialDataGenerator


def generate_features():
    match_results_file = osp.join(data_dir_path, source_dir, 'match_results.csv')
    match_results_df = pd.read_csv(match_results_file)
    feature_generator = FeatureGenerator(individual_window_size, head_to_head_window_size)
    features = feature_generator(match_results_df[-last_n_data:])
    features.to_csv(osp.join(data_dir_path, 'features_last_' + str(last_n_data) + '_windows_' + str(
        individual_window_size) + '_' + str(head_to_head_window_size) + '.csv'), index=False)


def generate_artificial_matches():
    try:
        artificial_matches_df = pd.read_csv(osp.join(data_dir_path, 'artificial_matches.csv'))
    except FileNotFoundError:
        print('Generating artificial matches')
        artificial_data_generator = ArtificialDataGenerator(individual_window_size, head_to_head_window_size)
        artificial_matches_df = artificial_data_generator.generate_artificial_matches(multiprocessing=False)
        print('Artificial matches details -', artificial_matches_df.describe())
        artificial_matches_df.to_csv(osp.join(data_dir_path, 'artificial_matches.csv'), index=False)
        print('Artificial matches saved to', osp.join(data_dir_path, 'artificial_matches.csv'))
    feature_generator = FeatureGenerator(individual_window_size, head_to_head_window_size)
    artificial_matches_df = artificial_matches_df.sample(frac=0.15).reset_index(drop=True)
    artificial_features = feature_generator(artificial_matches_df)
    artificial_features.to_csv(osp.join(data_dir_path, 'artificial_features_last_' + str(last_n_data) + '_windows_'
                                        + str(individual_window_size) + '_' + str(head_to_head_window_size) + '.csv'),
                               index=False)
