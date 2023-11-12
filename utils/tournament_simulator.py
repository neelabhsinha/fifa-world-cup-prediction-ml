#!/usr/bin/env python
# coding: utf-8


import numpy as np
import datetime
from const import WCGroups


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

class TournamentSimulator:

    def __init__(self, tournamentStartDate) -> None:
        self.tournamentStartDate = tournamentStartDate
        self.groups = WCGroups

    def predictWinner(self, team1, team2, dateOfMatch):
        winner = team1 if np.random.random(1) > 0.5 else team2
        print(f"{team1} v/s {team2} on {dateOfMatch}- {winner} won")
        return winner

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
                    winner = self.predictWinner(teams[i], teams[j], dateOfMatch)
                    winners.append(winner)
            u, count = np.unique(winners, return_counts=True)
            count_sort_ind = np.argsort(-count)
            grp_winners.append(u[count_sort_ind][:2])

        return np.array(grp_winners)

    def play(self, previousWinners, stage='round_of_16', start_date=datetime.date(2023, 1, 10)):
        winners = []
        i = 0
        print(f"\nStage: {stage}")
        while i < len(previousWinners):
            if stage == 'round_of_16':
                opp = i + 1 if i % 2 != 0 else i + 3
            else:
                opp = i + 1
            dateOfMatch = start_date + datetime.timedelta(days=i)
            winner = self.predictWinner(previousWinners[i], previousWinners[opp], dateOfMatch)
            winners.append(winner)
            if stage == 'round_of_16':
                i += 1 if i % 2 == 0 else 3
            else:
                i += 2
        print(f'\n{stage} winners: {winners}')
        return winners, dateOfMatch

    def playKnockOuts(self):
        groupStage = self.playGroupStage(start_date=self.tournamentStartDate)
        winners = groupStage.flatten()
        print(groupStage)
        stages = ['round_of_16', 'quarter_final', 'semi_final', 'final']
        last_date = self.tournamentStartDate + datetime.timedelta(days=55)
        for stage in stages:
            winners, last_date = self.play(winners, stage=stage, start_date=last_date + datetime.timedelta(days=3))
        return winners
