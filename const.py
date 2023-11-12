import numpy as np

# ---- Change the following paths as per your system ---- #
data_dir_path = './data'
project_dir_path = '.'
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

models = ['random_forest', 'logistic_regression', 'gradient_boost', 'support_vector_machine', 'decision_tree']

# ----Feature Selection Hyperparameters---- #
individual_window_size = 15
head_to_head_window_size = 15
last_n_data = 44000
pca_n_components = 5

# ----Hyperparameter Search Space for Random Forest---- #
random_forest_params = {'n_estimators': [int(x) for x in np.linspace(start=1, stop=25, num=2)],
                        'min_samples_split': [int(x) for x in np.linspace(5, 30)],
                        'min_samples_leaf': [int(x) for x in np.linspace(5, 30, num=1)],
                        'max_depth': [int(x) for x in np.linspace(1, 20, num=2)],
                        'max_features': ['log2', 'sqrt'],
                        'bootstrap': [True]
                        }

# ----Hyperparameter Search Space for Logistic Regression ---- #
logistic_regression_params = {
    'solver' : ['newton-cg','lbfgs','liblinear','sag','saga','newton-cholesky'],
    'penalty' : ['l2'],
    'C' : [100, 10, 1.0, 0.1, 0.01],
}

gradient_boost_params = {
    'n_estimators': [int(x) for x in np.linspace(1, 25, 2)],
    'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    'max_depth': [int(x) for x in np.linspace(1, 20, 2)],
    'min_samples_split': [int(x) for x in np.linspace(5, 30)],
    'min_samples_leaf': [int(x) for x in np.linspace(5, 30, num=1)],
    'subsample': np.arange(0.5, 1.0, 0.05),
}

# ----Hyperparameter Search Space for SVM ---- #
svm_param_distributions = {
    'C': np.logspace(-5, -3, 20),
    'gamma': np.logspace(-5, -1, 20),
    'kernel': ['linear', 'rbf'],
    "probability": [True]
}

# ----Hyperparameter Search Space for Decision Tree ---- #
decision_tree_params = {
    'max_depth': [None] + list(range(1, 200)),
    'min_samples_split': range(1, 30),
    'min_samples_leaf': range(1, 30),
    'criterion': ['gini', 'entropy']
}

n_iters = 200
cv = 5

# FIFA 2022 Groups
WCGroups = [
    ['Qatar', 'Ecuador', 'Senegal', 'Netherlands'],
    ['England', 'Iran', 'USA', 'Wales'],
    ['Argentina', 'Saudi Arabia', 'Mexico', 'Poland'],
    ['France', 'Australia', 'Denmark', 'Tunisia'],
    ['Spain', 'Costa Rica', 'Germany', 'Japan'],
    ['Belgium', 'Canada', 'Morocco', 'Croatia'],
    ['Brazil', 'Serbia', 'Switzerland', 'Cameroon'],
    ['Portugal', 'Ghana', 'Uruguay', 'South Korea']
]
