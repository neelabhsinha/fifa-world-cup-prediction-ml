import os.path as osp
import pandas as pd
import numpy as np

from const import data_dir_path, source_dir, features_keys, label_keys


class FeatureGenerator:
    def __init__(self, individual_window_size, head_to_head_window_size):
        match_results_file = osp.join(data_dir_path, source_dir, 'match_results.csv')
        self._match_results_df = pd.read_csv(match_results_file)
        rankings_file = osp.join(data_dir_path, source_dir, 'rankings.csv')
        self._rankings_df = pd.read_csv(rankings_file)
        self._match_results_df['date'] = pd.to_datetime(self._match_results_df['date'])
        self._rankings_df['rank_date'] = pd.to_datetime(self._rankings_df['rank_date'])
        self._individual_window_size = individual_window_size
        self._head_to_head_window_size = head_to_head_window_size

    def __call__(self, matches_df):
        features_labels_df = matches_df.apply(self._process_row, axis=1)
        features_labels_df.columns = features_keys + label_keys
        return features_labels_df

    def _process_row(self, row):
        home_team, away_team, match_date = row['home_team'], row['away_team'], row['date']
        home_goals, away_goals, neutral = row['home_score'], row['away_score'], row['neutral']
        features = self.get_features(home_team, away_team, match_date, neutral)
        labels = self.get_labels(home_goals, away_goals)
        return pd.Series(np.concatenate((features.flatten(), labels.flatten())))

    def get_features(self, home_team, away_team, match_date, neutral=False):
        home_mean_goals_scored_home, home_mean_goals_conceded_home, home_std_goals_scored_home, home_std_goals_conceded_home = self._get_individual_statistics(
            home_team, match_date, True)
        home_mean_goals_scored_away, home_mean_goals_conceded_away, home_std_goals_scored_away, home_std_goals_conceded_away = self._get_individual_statistics(
            home_team, match_date,False)
        away_mean_goals_scored_home, away_mean_goals_conceded_home, away_std_goals_scored_home, away_std_goals_conceded_home = self._get_individual_statistics(
            away_team, match_date, True)
        away_mean_goals_scored_away, away_mean_goals_conceded_away, away_std_goals_scored_away, away_std_goals_conceded_away = self._get_individual_statistics(
            away_team, match_date, False)
        h2h_home_team_mean_goals_home, h2h_away_team_mean_goals_away, h2h_home_team_std_goals_home, h2h_away_std_goals_away = self._get_head_to_head_statistics(
            home_team, away_team, match_date)
        h2h_home_team_mean_goals_away, h2h_away_team_mean_goals_home, h2h_home_team_std_goals_away, h2h_away_std_goals_home = self._get_head_to_head_statistics(
            away_team, home_team, match_date)
        home_rank, home_rank_change = self._get_closest_ranking_for_team(home_team, match_date)
        away_rank, away_rank_change = self._get_closest_ranking_for_team(away_team, match_date)
        home_win_count_home = self._get_home_win_count(home_team, match_date)
        home_win_count_away = self._get_away_win_count(home_team, match_date)
        away_win_count_home = self._get_home_win_count(away_team, match_date)
        away_win_count_away = self._get_away_win_count(away_team, match_date)
        h2h_home_win_count, h2h_away_win_count = self._get_head_to_head_win_count(home_team, away_team, match_date)
        if neutral:
            home_match_for_home_team = 0
        else:
            home_match_for_home_team = 1
        features_list = [home_win_count_home, home_win_count_away, away_win_count_home, away_win_count_away,
                         home_mean_goals_scored_home, home_mean_goals_conceded_home, home_std_goals_scored_home,
                         home_std_goals_conceded_home,
                         home_mean_goals_scored_away, home_mean_goals_conceded_away, home_std_goals_scored_away,
                         home_std_goals_conceded_away,
                         away_mean_goals_scored_home, away_mean_goals_conceded_home, away_std_goals_scored_home,
                         away_std_goals_conceded_home,
                         away_mean_goals_scored_away, away_mean_goals_conceded_away, away_std_goals_scored_away,
                         away_std_goals_conceded_away,
                         h2h_home_team_mean_goals_home, h2h_away_team_mean_goals_away, h2h_home_team_std_goals_home,
                         h2h_away_std_goals_away,
                         h2h_home_team_mean_goals_away, h2h_away_team_mean_goals_home, h2h_home_team_std_goals_away,
                         h2h_away_std_goals_home,
                         home_rank, home_rank_change, away_rank, away_rank_change, home_match_for_home_team]
        features = np.array(features_list)[:, np.newaxis].T
        features = np.nan_to_num(features)
        return features

    def get_labels(self, home_score, away_score):
        label_list = []
        if home_score > away_score:
            label_list = 1, 0, 0
        elif home_score < away_score:
            label_list = 0, 1, 0
        else:
            label_list = 0, 0, 1
        return np.array(label_list)[:, np.newaxis].T

    def _get_individual_statistics(self, team, date_of_new_match, home_team):
        self._match_results_df['date'] = pd.to_datetime(self._match_results_df['date'])
        matches_before_date = self._match_results_df[self._match_results_df['date'] < pd.to_datetime(date_of_new_match)]
        home_team = 'home_team' if home_team else 'away_team'
        home_score = 'home_score' if home_team else 'away_score'
        away_score = 'away_score' if home_team else 'home_score'
        team_a_matches = matches_before_date[(matches_before_date[home_team] == team)].tail(self._individual_window_size)
        if team_a_matches.empty:
            return 0, 0, 0, 0
        goals_scored = team_a_matches[home_score]
        goals_conceded = team_a_matches[away_score]
        mean_goals_scored = goals_scored.mean()
        mean_goals_conceded = goals_conceded.mean()
        std_goals_scored = goals_scored.std()
        std_goals_conceded = goals_conceded.std()
        return mean_goals_scored, mean_goals_conceded, std_goals_scored, std_goals_conceded

    def _get_head_to_head_statistics(self, home_team, away_team, date_of_new_match):
        self._match_results_df['date'] = pd.to_datetime(self._match_results_df['date'])
        matches_before_date = self._match_results_df[
            (self._match_results_df['date'] < pd.to_datetime(date_of_new_match)) &
            ((self._match_results_df['home_team'] == home_team) & (
                    self._match_results_df['away_team'] == away_team))]
        head_to_head_matches = matches_before_date.tail(self._head_to_head_window_size)
        if head_to_head_matches.empty:
            return 0, 0, 0, 0
        mean_goals_home = head_to_head_matches['home_score'].mean()
        mean_goals_away = head_to_head_matches['away_score'].mean()
        std_goals_home = head_to_head_matches['home_score'].std()
        std_goals_away = head_to_head_matches['away_score'].std()
        return mean_goals_home, mean_goals_away, std_goals_home, std_goals_away

    def _get_closest_ranking_for_team(self, team_name, target_date):
        self._rankings_df['rank_date'] = pd.to_datetime(self._rankings_df['rank_date'])
        target_date = pd.to_datetime(target_date)
        team_df = self._rankings_df[
            (self._rankings_df['country_full'].str.lower() == team_name.lower()) & (
                        self._rankings_df['rank_date'] <= target_date)]
        if team_df.empty:
            return 0, 0
        closest_rank_date = team_df['rank_date'].max()
        closest_rank_row = team_df[team_df['rank_date'] == closest_rank_date].iloc[0]
        return closest_rank_row['rank'], closest_rank_row['rank_change']

    def _get_home_win_count(self, team, date_of_new_match):
        matches_before_date = self._match_results_df[
            (self._match_results_df['date'] < pd.to_datetime(date_of_new_match)) &
            (self._match_results_df['home_team'] == team)
            ]
        home_wins = matches_before_date[matches_before_date['home_score'] > matches_before_date['away_score']].shape[0]
        return home_wins

    def _get_away_win_count(self, team, date_of_new_match):
        matches_before_date = self._match_results_df[
            (self._match_results_df['date'] < pd.to_datetime(date_of_new_match)) &
            (self._match_results_df['away_team'] == team)
            ]
        away_wins = matches_before_date[matches_before_date['away_score'] > matches_before_date['home_score']].shape[0]
        return away_wins

    def _get_head_to_head_win_count(self, home_team, away_team, date_of_new_match):
        matches_before_date = self._match_results_df[
            (self._match_results_df['date'] < pd.to_datetime(date_of_new_match)) &
            ((self._match_results_df['home_team'] == home_team) & (self._match_results_df['away_team'] == away_team))
            ]
        home_team_wins = matches_before_date[matches_before_date['home_score'] > matches_before_date['away_score']].shape[0]
        away_team_wins = matches_before_date[matches_before_date['home_score'] < matches_before_date['away_score']].shape[0]
        return home_team_wins, away_team_wins
