# ---- Change the following paths as per your system ---- #
data_dir_path = '/Users/neelabh/Desktop/Work/Projects/CS 7641 ML/data'

# ----Not needed to be changed---- #
source_dir = 'source'

features_keys = [
    'home_team_mean_goals_scored_at_home', 'home_team_mean_goals_conceded_at_home', 'home_team_std_goals_scored_at_home',
    'home_team_std_goals_conceded_at_home', 'home_team_mean_goals_scored_away', 'home_team_mean_goals_conceded_away',
    'home_team_std_goals_scored_away', 'home_team_std_goals_conceded_away', 'away_team_mean_goals_scored_at_home',
    'away_team_mean_goals_conceded_at_home', 'away_team_std_goals_scored_at_home', 'away_team_std_goals_conceded_at_home',
    'away_team_mean_goals_scored_away', 'away_team_mean_goals_conceded_away', 'away_team_std_goals_scored_away',
    'away_team_std_goals_conceded_away', 'h2h_home_team_mean_goals_scored_at_home', 'h2h_away_team_mean_goals_scored_away',
    'h2h_home_team_std_goals_scored_at_home', 'h2h_away_team_std_goals_scored_away', 'h2h_home_team_mean_goals_scored_away',
    'h2h_away_team_mean_goals_scored_at_home', 'h2h_home_team_std_goals_scored_away', 'h2h_away_team_std_goals_scored_at_home',
    'home_team_rank', 'home_team_rank_change', 'away_team_rank', 'away_team_rank_change', 'home_match_for_home_team'
]

label_keys = ['home_team_win', 'draw', 'away_team_win']

# ----Tuned Hyperparameters---- #
