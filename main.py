import argparse
import datetime
from const import models, data_dir_path, individual_window_size, head_to_head_window_size
from utils.preprocess import generate_features
from utils.tournament_simulator import TournamentSimulator

from utils.train import train


def arg_parser():
    parser = argparse.ArgumentParser(description='FIFA World Cup Prediction - CS 7641 Machine Learning Project for '
                                                 'Group 36, Fall 2023')
    parser.add_argument('--task', type=str, default=None, help='Perform a task', choices=['train', 'preprocess'])
    parser.add_argument('--model', type=str, default=None, help='Model to use for training and prediction',
                        choices=models)
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters for the model')
    parser.add_argument('--do_pca', action='store_true', help='Perform PCA on the data')
    parser.add_argument('--select_features', action='store_true', help='Generate features from the data')
    parser.add_argument('--simulate_tournament', action='store_true', help='Simulate a tournament')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    if args.task == 'train':
        train(args.model, args.do_pca, args.tune, args.select_features)
    elif args.task == 'preprocess':
        generate_features()
    elif args.simulate_tournament:
        tournamentSimulator= TournamentSimulator(datetime.date(2022,11,7))
        winner= tournamentSimulator.playKnockOuts()
        print(winner[0])
    