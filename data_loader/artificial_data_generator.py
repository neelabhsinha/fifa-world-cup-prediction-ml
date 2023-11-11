import os.path as osp
import os
import pandas as pd

from const import data_dir_path, source_dir
from concurrent.futures import ProcessPoolExecutor
from data_loader.feature_generator import FeatureGenerator


class ArtificialDataGenerator:
    def __init__(self, individual_batch_size, head_to_head_batch_size):
        self._feature_generator = FeatureGenerator(individual_batch_size, head_to_head_batch_size)
        match_results_file = osp.join(data_dir_path, source_dir, 'match_results.csv')
        match_results_df = pd.read_csv(match_results_file)
        self._match_results_df = self._feature_generator.get_world_cup_matches(match_results_df)
        self._match_results_df['date'] = pd.to_datetime(self._match_results_df['date'])

    def generate_artificial_matches(self, multiprocessing=True):
        dates = self._match_results_df['date'].unique()
        num_cores = os.cpu_count() if multiprocessing else 1
        chunk_size = len(dates) // num_cores
        date_chunks = [dates[i:i + chunk_size] for i in range(0, len(dates), chunk_size)]
        all_teams = set(self._match_results_df['home_team']).union(set(self._match_results_df['away_team']))
        if multiprocessing:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                results = executor.map(lambda chunk: self._generate_matches(chunk, all_teams), date_chunks)
            new_rows = [row for result in results for row in result]
        else:
            new_rows = self._generate_matches(date_chunks, all_teams)
        new_df = pd.DataFrame(new_rows, columns=self._match_results_df.columns)
        return new_df

    def _generate_matches(self, date_range, all_teams):
        artificial_matches = []
        for date in date_range[0]:
            date = pd.to_datetime(date)
            matches_before = self._match_results_df[(self._match_results_df['date'] < date)]
            existing_head_to_head = set(
                tuple(sorted([home, away]))
                for home, away in zip(matches_before['home_team'], matches_before['away_team'])
            )
            daily_teams = (set(self._match_results_df[self._match_results_df['date'] == date]['home_team'])
                           .union(set(self._match_results_df[self._match_results_df['date'] == date]['away_team'])))
            for team_a in daily_teams:
                for team_b in (all_teams - daily_teams):
                    match_pair = tuple(sorted([team_a, team_b]))
                    if match_pair in existing_head_to_head:
                        original_row = self._match_results_df[((self._match_results_df['date'] == date) & (
                            (self._match_results_df['away_team'] == team_a)
                            | (self._match_results_df['home_team'] == team_a)))].iloc[0]
                        new_row = {
                            'date': date,
                            'home_team': team_a,
                            'away_team': team_b,
                            'home_score': -1,
                            'away_score': -1,
                            'tournament': original_row['tournament'],
                            'city': original_row['city'],
                            'country': original_row['country'],
                            'neutral': original_row['neutral']
                        }
                        artificial_matches.append(new_row)
        return artificial_matches

