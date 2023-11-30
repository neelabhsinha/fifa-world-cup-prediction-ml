#!/usr/bin/env python
# coding: utf-8


from feature.feature_generator import FeatureGenerator
from model.decision_tree import DecisionTree
from model.gmm import GaussianMixtureModel
from model.gradient_boost import GradientBoost
from model.kmeans import Kmeans
from model.logistic_regression_class import LogisticRegressionClass
from model.random_forest import RandomForest
from model.svm import SVM
import numpy as np
import datetime
from const import WCGroups, individual_window_size, head_to_head_window_size
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import pandas as pd
from matplotlib import pyplot as plt

# ## Tournament Schedule
# 
# Total matches= 64
# 
# 1. Group Stage- 8 groups of 4 teams each<br>
#     Each team plays 3 matches with the other teams in the group<br>
#     Total matches per group= 6 (4C2)<br>
#     Total matches= 48<br>
# 
# Knockout Stages- Played after Group Stages
# 1. Round of 16- 8 Matches (16C2 Matches)<br>
#     First-place Group A vs. Second-place Group B- W1<br>
#     First-place Group B vs. Second-place Group A- W2<br>
#     First-place Group C vs. Second-place Group D- W3<br>
#     First-place Group D vs. Second-place Group C- W4<br>
#     First-place Group E vs. Second-place Group F- W5<br>
#     First-place Group F vs. Second-place Group E- W6<br>
#     First-place Group G vs. Second-place Group H- W7<br>
#     First-place Group H vs. Second-place Group G- W8<br>
# 
# 2. Quarter Finals- 4 Matches (8C2)<br>
#     W1 vs W2- QF_W1<br>
#     W3 vs W4- QF_W2<br>
#     W5 vs W6- QF_W3<br>
#     W7 vs W8- QF_W4<br>
# 
# 3. Semi Finals- 2 Matches (4C2)<br>
#     QF_W1 vs QF_W2- SF_W1<br>
#     QF_W3 vs QF_W4- SF_W2<br>
# 
# 4. Play-offs/ Third Place- 1 Match (2C2)<br> 
#     Semi Final Losers<br>
# 
# 5. Final- 1 Match<br>
#     SF_W1 vs SF_W2<br>
# 

class TournamentSimulator():

    def __init__(self, tournamentStartDate, model_name, unsupervised_model_name) -> None:
        self.tournamentStartDate= tournamentStartDate
        self.featureGenerator= FeatureGenerator(individual_window_size, head_to_head_window_size)
        self.model_name= model_name
        if model_name == 'random_forest':
            self.model = RandomForest()
        elif model_name == 'gradient_boost':
            self.model = GradientBoost()
        elif model_name == 'support_vector_machine':
            self.model = SVM()
        elif model_name == 'decision_tree':
            self.model = DecisionTree()
        elif model_name == 'logistic_regression':
            self.model = LogisticRegressionClass()
        if unsupervised_model_name == 'gmm':
            self.unsupervised_model= GaussianMixtureModel()
        elif unsupervised_model_name == 'kmeans':
            self.unsupervised_model= Kmeans()
        self.groups= self.getGroups()
        self.model.load_model()

    def getGroups(self):
        all_countries= [element for sublist in WCGroups for element in sublist]
        features= [self.featureGenerator._get_individual_statistics_ranks(country, datetime.date(2023, 11, 1), None) for country in all_countries]
        clusters= self.unsupervised_model.get_clusters(features, all_countries)
        groups=np.array([ clusters[:,i] for i in range(8)])
        return groups

    def predictWinner(self, team1, team2, dateOfMatch, matchType):
        features= self.featureGenerator.get_features(team1, team2, dateOfMatch, matchType)
        win_prob= self.model.predict_proba(features)[0][1]
        winner= team1 if win_prob>0.5 else team2
        print(f"{team1} v/s {team2} on {dateOfMatch}- {winner} won")
        return winner, win_prob

    def playGroupStage(self, start_date):
        num_grps = 8
        grp_winners = []
        print(f"Groups for this tournament:\n{self.groups}\n\n")
        print('Group Stage:')
        for grp in range(num_grps):
            teams = self.groups[grp]
            winners = []
            for i in range(0, len(teams)):
                for j in range(i + 1, len(teams)):
                    dateOfMatch = start_date + datetime.timedelta(days=i * j)
                    winner, _= self.predictWinner(teams[i], teams[j], dateOfMatch, 'group_stage')
                    winners.append(winner)
            u, count = np.unique(winners, return_counts=True)
            count_sort_ind = np.argsort(-count)
            grp_winners.append(u[count_sort_ind][:2])

        return np.array(grp_winners)

    def play(self, previousWinners, stage='round_of_16', start_date=datetime.date(2023, 1, 10), labels=[], odds=[]):
        winners = []
        i = 0
        print(f"\nStage: {stage}")
        while i < len(previousWinners):
            if stage == 'round_of_16':
                opp = i + 1 if i % 2 != 0 else i + 3
            else:
                opp = i + 1
            dateOfMatch = start_date + datetime.timedelta(days=i)
            winner, win_prob = self.predictWinner(previousWinners[i], previousWinners[opp], dateOfMatch, stage)
            winners.append(winner)
            labels.append(f"{previousWinners[i]}({np.round(win_prob,2)}) vs. {previousWinners[opp]}({np.round(1-win_prob,2)})")
            odds.append([win_prob, 1-win_prob])
            if stage == 'round_of_16':
                i += 1 if i % 2 == 0 else 3
            else:
                i += 2
        print(f'\n{stage} winners: {winners}')
        return winners, dateOfMatch, labels, odds

    def playKnockOuts(self):
        groupStage = self.playGroupStage(start_date=self.tournamentStartDate)
        winners = groupStage.flatten()
        print(groupStage)
        stages = ['round_of_16', 'quarter_final', 'semi_final', 'final']
        last_date = self.tournamentStartDate + datetime.timedelta(days=55)
        labels= list()
        odds= list()
        for stage in stages:
            winners, last_date, labels, odds = self.play(winners, stage=stage, start_date=last_date + datetime.timedelta(days=3), labels=labels, odds=odds)
        return winners, labels, odds

    def visualizeKnockOuts(self):
        model_name= ' '.join(model.capitalize() for model in self.model_name.split('_'))
        winner, labels, odds= self.playKnockOuts()
        node_sizes = pd.DataFrame(list(reversed(odds)))
        scale_factor = 0.3 # for visualization
        G = nx.balanced_tree(2, 3)
        pos = graphviz_layout(G, prog='twopi')
        centre = pd.DataFrame(pos).mean(axis=1).mean()

        plt.figure(figsize=(15, 15))
        ax = plt.subplot(1,1,1)
        # add circles 
        circle_positions = [(235, 'black'), (180, 'blue'), (120, 'red'), (60, 'yellow')]
        [ax.add_artist(plt.Circle((centre, centre), 
                                cp, color='grey', 
                                alpha=0.2)) for cp, c in circle_positions]

        # draw first the graph
        nx.draw(G, pos, 
                node_color=node_sizes.diff(axis=1)[1].abs().pow(scale_factor), 
                # node_size=node_sizes.diff(axis=1)[1].abs().pow(scale_factor)*2000, 
                alpha=1, 
                cmap='Reds',
                edge_color='black',
                width=10,
                with_labels=False)

        # draw the custom node labels
        shifted_pos = {k:[(v[0]-centre)*0.9+centre,(v[1]-centre)*0.9+centre] for k,v in pos.items()}
        nx.draw_networkx_labels(G, 
                                pos=shifted_pos, 
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=.5, alpha=1), font_size=8,
                                labels=dict(zip(reversed(range(len(labels))), labels)))

        texts = ((10, 'Best 16', 'black'), (70, 'Quarter-\nfinal', 'blue'), (130, 'Semifinal', 'red'), (190, 'Final', 'yellow'))
        [plt.text(p, centre+20, t, 
                fontsize=12, color='grey', 
                va='center', ha='center') for p,t,c in texts]
        plt.axis('equal')
        plt.title(f'Single-elimination phase predictions with fair odds using {model_name}\nWinner= {winner[0]}', fontsize=15)
        plt.show()
