import numpy as np

# ---- Change the following paths as per your system ---- #
data_dir_path = '/Users/neelabh/Desktop/Work/Projects/CS 7641 ML/data'
project_dir_path = '/Users/neelabh/Desktop/Work/Projects/CS 7641 ML/code'
# ----Not needed to be changed---- #
source_dir = 'source'

features_keys = [
    "home_win_count", "away_win_count", "h2h_home_win_count", "h2h_away_win_count",
    "home_mean_goals_scored", "home_mean_goals_conceded", "home_std_goals_scored", "home_std_goals_conceded",
    "away_mean_goals_scored", "away_mean_goals_conceded", "away_std_goals_scored", "away_std_goals_conceded",
    "h2h_home_mean_goals", "h2h_away_mean_goals", "h2h_home_std_goals", "h2h_team_std_goals",
    "home_rank - away_rank", "home_rank_change", "away_rank_change",
    "home_team_opposition_rank_diff", "away_team_opposition_rank_diff",
    "is_friendly", "is_qualifier", "is_tournament", "home_match_for_home_team"
]

label_keys = ['home_team_win', 'draw', 'away_team_win']

models = ['random_forest', 'logistic_regression', 'gradient_boost']

# ----Feature Selection Hyperparameters---- #
individual_window_size = 15
head_to_head_window_size = 15
last_n_data = 44000
pca_n_components = 5

# ----Hyperparameter Search Space for Random Forest---- #
random_forest_params = {'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
                        'min_samples_split': [int(x) for x in np.linspace(1, 10)],
                        'min_samples_leaf': [int(x) for x in np.linspace(1, 10, num=1)],
                        'max_depth': [int(x) for x in np.linspace(10, 200, num=10)],
                        'max_features': ['log2', 'sqrt'],
                        'bootstrap': [True, False]
                        }

# ----Hyperparameter Search Space for Gradient Boosters ---- #
gradient_boost_params = {
    'n_estimators': [int(x) for x in np.linspace(100, 1000, 10)],
    'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    'max_depth': [int(x) for x in np.linspace(10, 200, 10)],
    'min_samples_split': [int(x) for x in np.linspace(1, 10)],
    'min_samples_leaf': [int(x) for x in np.linspace(1, 10, num=1)],
    'subsample': np.arange(0.5, 1.0, 0.05),
}

n_iters = 200
cv = 10
