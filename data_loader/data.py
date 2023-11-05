# This is the feature selector file .
#TODO : Delete unnecessary commented code.
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from const import *
import warnings
warnings.filterwarnings("ignore")
#   DataSetSelection is meant for your use .It works on generated csvs
# DataSetGeneration generates a fresh dataset for your work. Use it iff you wanna see the effects of changing hyperparameters.
# ArgParser is not planned yet to combine the two classes . Maybe at the end .
# Next is putting the feature selector in DataSetSelection to allow you the ability to set correlation thresholds.
# Parallely the new class to be added. On it .
#UNDER CONSTRUCTION
class DataSetSelection:  #Use this for your classifiers!
    def __init__(self):
        #Master dataset should be read here . It never changes. It is the csv with all features before truncation.
        #TODO : Paste all of the data reading code from the notebooks and test. Block - by - block ( Cell 10-13)
        # I am not sure if pandas dataset is passed
        self.pca = False
        self.lda=False
        self.corr = False
        self.featurelist = []
        self.dataset = pd.read_csv('data_loader/master_dataset.csv')
        self.fifarankdataset = pd.read_csv("data_loader/kaggle/input/fifa_ranking-2023-07-20.csv")
        self.masterdataset = 0
        #TODO : Check this


    def features_binary(self,teamA,teamB,date):
        """ UNDER CONSTRUCTION
          Returns team attributes for a match(Complete set of features right now .)
        You may use this if you want . This returns all of the original features appended. Supervised learners may wanna use
        supervised_dataset_final for their learning set. I assume the tournament folks would need it during simulation.
        This might take a long time to run. Absolutely don't do it at every match node .
        args :
                teamA : string that is  the Full Name of the first team 
                teamB:  string that is  the Full Name of the second team 
                date: DateTime format "YYYY-MM-DD" when you intend to face them off .
                timeWindow : (Not implemented yet . How far do you want to compare their strengths)
        returns :
                [team_A_unary_attributes|team_B_unary_attributes|team_AvB_data] or PCAd sphagetti            
        """
        # I have verified this data with explicit hand tracing with two random football teams over a period of time for various teams.
        # however in case I missed please let me know ASAP. If you find data which is not historically true ,that means the data is wrong in the master dataset as well. 
        #print(feature_candidates) 
        fifa_rank = self.fifarankdataset  # This is unavoidable. We need to know the ranks of the two teams on the date
        fifa_rank["rank_date"] = pd.to_datetime(fifa_rank["rank_date"]) #Fifa rank 
        fifa_rank["country_full"] = fifa_rank["country_full"].str.replace("IR Iran", "Iran").str.replace("Korea Republic", "South Korea").str.replace("USA", "United States")
        fifa_rank = fifa_rank.set_index(['rank_date']).groupby(['country_full'], group_keys=False).resample('D').first().fillna(method='ffill').reset_index()
        features_binary = []
        n_games= 5 # 5 games 
        n_historic_window = 5 #5 years # TODO: This needs to be hardcoded in some way.
        # Faster actually since you just need to two slices
        masterdataset = self.dataset
        #SLICE N1 :
        teams = [teamA,teamB]
        fifarankdate = fifa_rank[fifa_rank['rank_date']== date]
        rankA = fifarankdate[fifarankdate["country_full"] == teamA]
        rankA = rankA.iloc[0]["rank"]
        rankB = fifarankdate[fifarankdate["country_full"] == teamB]
        rankB = rankB.iloc[0]["rank"]
        del_rank = rankA - rankB
        stats_val = []
        stats_val0 = [del_rank]
        masterdataset["date"] = pd.to_datetime(masterdataset["date"])
        masterdataset = masterdataset[(masterdataset["date"] < date)].reset_index(drop=True) #Always filter by date first.
        date = pd.to_datetime(date)
        for team in teams:
            team_history =  masterdataset.loc[ (masterdataset["date"] >= date-n_historic_window*pd.Timedelta(days=365)) & ((masterdataset["home_team"] == team) | (masterdataset["away_team"] == team))].sort_values(by=['date'], ascending=False) #Always filter by date first.
            team_history=team_history.drop(["index"],axis=1)
            team_history = team_history.reset_index(drop=True)
            lastn = team_history.head(n_games)
            lastn = lastn.reset_index(drop=True)
            n_l = 0
            goals = 0
            goals_ln=0
            goals_suf = 0
            goals_suff_ln = 0
            rank_ln = 0
            gp_ln = 0
            gp = 0
            rank = 0
            n_l = lastn.shape[0]
            if n_l!=0:
                for i in range(n_l):  #I don't like pandas and don't want to.Accessing a cell is harder than anything
                    if lastn.iloc[i]["home_team"] == team:
                        goals_ln = goals_ln+lastn.iloc[i]["home_score"]
                        goals_suff_ln = goals_suff_ln+lastn.iloc[i]["away_score"]
                        gp_ln += lastn.iloc[i]["total_points_home"]
                        rank_ln += lastn.iloc[i]["rank_away"]
                    else:
                        goals_ln = goals_ln+lastn.iloc[i]["away_score"]
                        goals_suff_ln = goals_suff_ln+lastn.iloc[i]["home_score"]
                        gp_ln += lastn.iloc[i]["total_points_away"]
                        rank_ln += lastn.iloc[i]["rank_home"]

           
            if n_l ==0:
                #total_points_home 
                goals_ln = 0
                gp_ln = 0
                rank_ln = 0
                goals_suff_ln = 0
            else:
                goals_ln = goals_ln/n_l
                goals_suff_ln  = goals_suff_ln/n_l
                gp_ln = gp_ln/n_l
                rank_ln = rank_ln/n_l   
            if len(lastn) == 0:
                points_ln = 0
            else :
                t0 = lastn.head(1)
                
                z = t0[t0["home_team"] == team]
                if len(z) !=0 :
                    point_t0 = t0.iloc[0]["total_points_home"]
                else:
                    point_t0 = t0.iloc[0]["total_points_away"]
                tn = lastn.tail(1)
                z = tn[tn["home_team"] == team]
                if len(z)!=0 :
                    point_tn = tn.iloc[0]["total_points_home"]
                else:
                    point_tn = tn.iloc[0]["total_points_away"]             
                points_ln = point_tn - point_t0
            t0 = team_history.head(1)
            z = t0[t0["home_team"] == team]
            if len(z)!=0 :
                point_t0 = t0.iloc[0]["total_points_home"]
            else:
                point_t0 = t0.iloc[0]["total_points_away"]
            tn = team_history.tail(1)
            z = tn[tn["home_team"] == team]
            if  len(z)!=0:
                point_tn = tn.iloc[0]["total_points_home"]
            else:
                point_tn = tn.iloc[0]["total_points_away"]             
            points = point_tn - point_t0
            team_history_as_home =  masterdataset.loc[ (masterdataset["date"] >= date-n_historic_window*pd.Timedelta(days=365)) & ((masterdataset["home_team"] == team) )].sort_values(by=['date'], ascending=False)
            team_history_as_home=team_history_as_home.drop(["index"],axis=1)
            team_history_as_home = team_history_as_home.reset_index(drop=True)
            n_H = team_history_as_home.shape[0]
            goals = goals + team_history_as_home["home_score"].sum() 
            goals_suf = goals_suf + team_history_as_home["away_score"].sum()
            gp = gp+team_history_as_home["total_points_home"].sum()
            team_history_as_away =  masterdataset.loc[ (masterdataset["date"] >= date-n_historic_window*pd.Timedelta(days=365)) & ((masterdataset["away_team"] == team) )].sort_values(by=['date'], ascending=False)
            team_history_as_away=team_history_as_away.drop(["index"],axis=1)
            team_history_as_away = team_history_as_away.reset_index(drop=True)
            n_A = team_history_as_away.shape[0]
            goals = goals + team_history_as_away["away_score"].sum() 
            goals_suf = goals_suf + team_history_as_away["home_score"].sum()
            gp = gp+team_history_as_away["total_points_away"].sum()
            if (n_A+n_H)!=0:
                goals = goals/(n_A+n_H)
                goals_suf = goals_suf/(n_A+n_H)
                gp = gp/(n_A+n_H)
                rank = rank/(n_A+n_H)
            stats_val.append([goals, goals_ln, goals_suf, goals_suff_ln, rank, rank_ln,points,points_ln,gp,gp_ln])
        stats_val = stats_val[0]+stats_val[1]
        stats_val = stats_val0+stats_val
        n_historic_window_f2f = 5
        past_games_AvB = masterdataset[(masterdataset['home_team'] == teamA)&((masterdataset['away_team'] == teamB)) & (masterdataset["date"] < date) & (masterdataset['date'] > date- n_historic_window_f2f*pd.Timedelta(days=365) )].sort_values(by=['date'], ascending=False)
        n_AvB = past_games_AvB.shape[0]
        goals_by_A = 0
        goals_by_B = 0
        won_by_A =0 
        won_by_B = 0
        tied_AvB = 0
        goals_by_A = past_games_AvB["home_score"].sum() + goals_by_A
        goals_by_B = past_games_AvB["away_score"].sum() +goals_by_B
        won_by_A = past_games_AvB["Target_Outcome_Win"].sum() +won_by_A
        won_by_B = past_games_AvB["Target_Outcome_Loss"].sum() + won_by_B
        tied_AvB = past_games_AvB["Target_Outcome_Tie"].sum() +tied_AvB
        past_games_BvA =masterdataset[(masterdataset['home_team'] == teamB)&((masterdataset['away_team'] == teamA)) & (masterdataset["date"] < date) & (masterdataset['date'] > date- n_historic_window_f2f*pd.Timedelta(days=365) )].sort_values(by=['date'], ascending=False)
        n_BvA = past_games_BvA.shape[0]
        goals_by_A = past_games_BvA["away_score"].sum() + goals_by_A
        goals_by_B = past_games_BvA["home_score"].sum() +goals_by_B
        won_by_A = past_games_BvA["Target_Outcome_Loss"].sum() +won_by_A
        won_by_B = past_games_BvA["Target_Outcome_Win"].sum() + won_by_B
        tied_AvB = past_games_BvA["Target_Outcome_Tie"].sum() +tied_AvB
        if (n_AvB + n_BvA)!=0:
            goals_by_A = goals_by_A/(n_AvB+n_BvA)
            goals_by_B = goals_by_B/(n_AvB+n_BvA)
        stats_val = stats_val + [goals_by_A,goals_by_B,won_by_A,won_by_B,tied_AvB]
        stats_cols= ['del_rank','home_goals_mean','home_goals_mean_ln','home_goals_suf_mean','home_goals_suf_mean_ln', 'home_rank_mean',
       'home_rank_mean_ln', 'home_points', 'home_points_ln',
       'home_game_points_mean', 'home_game_points_mean_ln', 
       'away_goals_mean', 'away_goals_mean_ln', 'away_goals_suf_mean',
       'away_goals_suf_mean_ln', 'away_rank_mean', 'away_rank_mean_ln',
       'away_points', 'away_points_ln', 'away_game_points_mean',
       'away_game_points_mean_ln', 'goals_AvB_past', 'goals_BvA_past',"A_victories","B_victories","Team_ties"]
        features_binary =  pd.DataFrame([stats_val], columns=stats_cols)
        #return 0
        return features_binary
    def supervised_dataset_final(self,dimred_method = "Corr",date_start ='2000-01-01',date_end = '2023-01-01'):
        """ UNDER CONSTRUCTION
        This is the final reduced dataset we will work with.
        args:
            dimrec_medthod = Method for dimension reduction 
                                'PCA' for PCA
                                'LDA' for LDA (Not Implemented yet)
                                'PCA+LDA' PCA followed by LDA (Not Implemented yet.Don't count on it.)
                                'Corr' for Correlation reduced dataset
                                'None' for full dataset
            date_start = Date from which you'll consider the data relevant(YYYY-MM-DD)
            date_end = Date till which you'll consider the data relevant(YYYY-MM-DD)
                                
        returns : (pandas [X_data],[Y_data]
                                pandas_dataframe :Pandas dataframe for your data to validate,test and train
                                feature_list : An array of strings containing column names . Use this to filter the dataset
                                    Convert the pandas array to numpy after extracting   
                                target_list: One hot coded output array names                                              """
        if date_start >= date_end:
            raise ValueError('Start date cannot be after or the same as end')
        if dimred_method not in ["None","Corr",'PCA','LDA']:
            raise ValueError('Not a valid dimension reudction method')
        if dimred_method == "None":
            self.pca = False
            self.lda=False
            self.corr = False
            masterdataset=pd.read_csv('data_loader/master_dataset_automated.csv')
            masterdataset["date"] = pd.to_datetime(masterdataset["date"])
            masterdataset["del_rank"] = masterdataset["rank_home"] - masterdataset["rank_away"]
            self.masterdataset = masterdataset
            masterdataset = masterdataset[(masterdataset["date"] >= date_start) & (masterdataset["date"] < date_end)].reset_index(drop=True) #Always filter by date first.
            featureset = ["neutral","tournament","rank_home","rank_away", "del_rank",'home_goals_mean','home_goals_mean_ln','home_goals_suf_mean','home_goals_suf_mean_ln', 'home_rank_mean',
            'home_rank_mean_ln', 'home_points', 'home_points_ln',
            'home_game_points_mean', 'home_game_points_mean_ln', 
            'away_goals_mean', 'away_goals_mean_ln', 'away_goals_suf_mean',
            'away_goals_suf_mean_ln', 'away_rank_mean', 'away_rank_mean_ln',
            'away_points', 'away_points_ln', 'away_game_points_mean',
            'away_game_points_mean_ln', 'goals_AvB_past', 'goals_BvA_past',
            'A_victories', 'B_victories', 'Team_ties']
            targetset = ["Target_Outcome_Win","Target_Outcome_Loss","Target_Outcome_Tie"]
            self.featurelist = featureset
            return masterdataset[featureset],masterdataset[targetset]
        if dimred_method == "Corr": # This is the default behavior
            self.pca = False
            self.lda=False
            self.corr = True   
            masterdataset=pd.read_csv("data_loader/master_dataset_automated.csv")
            masterdataset["date"] = pd.to_datetime(masterdataset["date"]) 
            masterdataset["del_rank"] = masterdataset["rank_home"] - masterdataset["rank_away"]
            masterdataset = masterdataset[(masterdataset["date"] >= date_start) & (masterdataset["date"] < date_end)].reset_index(drop=True) #Always filter by date first.
            targetset = ["Target_Outcome_Win","Target_Outcome_Loss","Target_Outcome_Tie"]
            featureset = ["tournament","rank_home","rank_away","del_rank",'home_goals_mean','home_goals_mean_ln','home_goals_suf_mean', 
            'away_goals_mean', 'away_goals_mean_ln', 'away_goals_suf_mean','goals_AvB_past', 'goals_BvA_past',
            'A_victories', 'B_victories']
            self.featurelist = featureset
            return masterdataset[featureset],masterdataset[targetset]
        if dimred_method == "PCA":
            self.pca = True
            self.lda=False
            self.corr = False
            masterdataset=pd.read_csv('data_loader/PCAdData.csv')
            masterdataset["date"] = pd.to_datetime(masterdataset["date"]) 
            masterdataset = masterdataset[(masterdataset["date"] >= date_start) & (masterdataset["date"] < date_end)].reset_index(drop=True) #Always filter by date first.
            targetset = ["Target_Outcome_Win","Target_Outcome_Loss","Target_Outcome_Tie"]   
            featureset = ['X_Principal0','X_Principal1','X_Principal2','XPrincipal3']
            self.featurelist = featureset
            return masterdataset[featureset],masterdataset[targetset] 
        
        #raise "NotImplementedError"    
    # Ignore unsupervised for now
    def unsupervised_features_final(self,pca="False"):
        """ UNDER CONSTRUCTION"""
        print("sorry under construction!")
        return 0
class DataSetGeneration:  # This generated the original Data. Use this if you need to perform hyperparameter tuning.
    def __init__(self) -> None:
        pass
    # TODO: Dump Jupyter Code here. 
    def generate_unary_team_histoy(self,team_stats,n_historic_window,n_games):
        stats_val = []
        for index, row in tqdm(team_stats.iterrows()):
            # 1.5 minutes runtime
            team = row["team"]
            date = row["date"]
            past_games = team_stats.loc[(team_stats["team"] == team) & (team_stats["date"] < date) & (team_stats['date'] > date- n_historic_window*pd.Timedelta(days=365) )].sort_values(by=['date'], ascending=False)
            lastn = past_games.head(n_games)
            
            goals = past_games["score"].mean()
            goals_ln = lastn["score"].mean()
            
            goals_suf = past_games["suf_score"].mean()
            goals_suf_ln = lastn["suf_score"].mean()
            
            rank = past_games["rank_suf"].mean()
            rank_ln = lastn["rank_suf"].mean()
            
            if len(lastn) > 0:
                points = past_games["total_points"].values[0] - past_games["total_points"].values[-1]
                points_ln = lastn["total_points"].values[0] - lastn["total_points"].values[-1] 
            else:
                points = 0
                points_ln = 0
                
            gp = past_games["total_points"].mean()
            gp_ln = lastn["total_points"].mean()
            

            
            stats_val.append([goals, goals_ln, goals_suf, goals_suf_ln, rank, rank_ln,points,points_ln,gp,gp_ln])
        stats_cols = ["goals_mean", "goals_mean_ln", "goals_suf_mean", "goals_suf_mean_ln", "rank_mean", "rank_mean_ln","points","points_ln",  "game_points_mean", "game_points_mean_ln"]

        stats_df = pd.DataFrame(stats_val, columns=stats_cols)
        return stats_df
    def generate_masterdataset(self):
        # Datasets
        #1870s football data
        historic_matches = pd.read_csv('data_loader/kaggle/input/results.csv')
        #45100 matches
        #Fifa Rankinf Data
        fifa_rank = pd.read_csv('data_loader/kaggle/input/fifa_ranking-2023-07-20.csv')
        # Let's drop all games before 1990 (fifa rank data starts here.You have 28k matches now)
        historic_matches["date"] = pd.to_datetime(historic_matches["date"]) #Fixing non-sense dates
        fifa_rank["rank_date"] = pd.to_datetime(fifa_rank["rank_date"]) #Fixing non-sense dates
        historic_matches = historic_matches[(historic_matches["date"] >= "1990-1-1")].reset_index(drop=True)
        #Let's start pasting this data
        #Change Names
        fifa_rank["country_full"] = fifa_rank["country_full"].str.replace("IR Iran", "Iran").str.replace("Korea Republic", "South Korea").str.replace("USA", "United States")
        # We first add two kinds of columns one for home_team and one for away_team 
        fifa_rank = fifa_rank.set_index(['rank_date']).groupby(['country_full'], group_keys=False).resample('D').first().fillna(method='ffill').reset_index()
        historic_matches_ranked = historic_matches.merge(fifa_rank[["country_full", "total_points", "previous_points", "rank", "rank_change", "rank_date"]], left_on=["date", "home_team"], right_on=["rank_date", "country_full"]).drop(["rank_date", "country_full"], axis=1)
        historic_matches_ranked = historic_matches_ranked.merge(fifa_rank[["country_full", "total_points", "previous_points", "rank", "rank_change", "rank_date"]], left_on=["date", "away_team"], right_on=["rank_date", "country_full"], suffixes=("_home", "_away")).drop(["rank_date", "country_full"], axis=1)
        #28619 matches
        #Tournament coded as friendly/non_friendly ( 0 and 1)
        # This is an interesting feature. I am not convinced it should be used.
        historic_matches_ranked["tournament"] = historic_matches_ranked.apply(lambda row : 0 if (row.tournament == "Friendly") else 1 ,axis=1)
        # One-hot coded label generation
        historic_matches_ranked['GoalDifference'] = historic_matches_ranked.apply(lambda row: row.home_score - row.away_score,axis=1)
        historic_matches_ranked['Target_Outcome_Win'] = historic_matches_ranked.apply(lambda row : 1 if (row.GoalDifference > 0) else 0 ,axis=1)
        historic_matches_ranked['Target_Outcome_Loss'] = historic_matches_ranked.apply(lambda row : 1 if (row.GoalDifference < 0) else 0 ,axis=1)
        historic_matches_ranked['Target_Outcome_Tie'] = historic_matches_ranked.apply(lambda row : 1 if (row.GoalDifference == 0) else 0 ,axis=1)
        historic_matches_ranked["del_rank"] = historic_matches_ranked["rank_home"] - historic_matches_ranked["rank_away"]
        #Team wise stats calculator
        #Add new features derived from historical match data
        # Goals taken and scored against in the last n matches (?)
        # Unary features
        A_team =historic_matches_ranked[["date", "home_team","home_score","away_score","rank_home", "rank_away","rank_change_home", "total_points_home", "Target_Outcome_Win","Target_Outcome_Loss","Target_Outcome_Tie"]]

        B_team = historic_matches_ranked[["date", "away_team","home_score","away_score",  "rank_away", "rank_home","rank_change_away", "total_points_away", "Target_Outcome_Win","Target_Outcome_Loss","Target_Outcome_Tie"]]
        A_team.columns = [h.replace("home_", "").replace("_home", "").replace("away_", "suf_").replace("_away", "_suf") for h in A_team.columns]
        B_team.columns = [a.replace("away_", "").replace("_away", "").replace("home_", "suf_").replace("_home", "_suf") for a in B_team.columns]
        team_stats = pd.concat([A_team,B_team],ignore_index=True)
        stats_df = self.generate_unary_team_histoy(team_stats,5,5)
        full_df = pd.concat([team_stats.reset_index(drop=True), stats_df], axis=1, ignore_index=False)
        A_team_stats = full_df.iloc[:int(full_df.shape[0]/2),:]
        B_team_stats = full_df.iloc[int(full_df.shape[0]/2):,:]
        A_team_stats = A_team_stats[A_team_stats.columns[-10:]]
        B_team_stats = B_team_stats[B_team_stats.columns[-10:]]
        A_team_stats.columns = ['home_'+str(col) for col in A_team_stats.columns]
        B_team_stats.columns = ['away_'+str(col) for col in B_team_stats.columns]
        match_stats = pd.concat([A_team_stats, B_team_stats.reset_index(drop=True)], axis=1, ignore_index=False)
        detailed_historical_matches_ranked = pd.concat([historic_matches_ranked, match_stats.reset_index(drop=True)], axis=1, ignore_index=False)
        stats_val = []
        n_historic_window_f2f = 5 #5 years
        #n_games = 5 #Number of games to consider
        for index, row in tqdm(detailed_historical_matches_ranked.iterrows()):
            #1.5 minutes runtime
            teamA = row["home_team"]
            teamB = row["away_team"]
            date = row["date"]
            past_games_AvB = detailed_historical_matches_ranked[(detailed_historical_matches_ranked['home_team'] == teamA)&((detailed_historical_matches_ranked['away_team'] == teamB)) & (detailed_historical_matches_ranked["date"] < date) & (detailed_historical_matches_ranked['date'] > date- n_historic_window_f2f*pd.Timedelta(days=365) )].sort_values(by=['date'], ascending=False)
            #lastn = past_games.head(n_games)
            n_AvB = past_games_AvB.shape[0]
            goals_by_A = 0
            goals_by_B = 0
            won_by_A =0 
            won_by_B = 0
            tied_AvB = 0
            goals_by_A = past_games_AvB["home_score"].sum() + goals_by_A
            goals_by_B = past_games_AvB["away_score"].sum() +goals_by_B
            won_by_A = past_games_AvB["Target_Outcome_Win"].sum() +won_by_A
            won_by_B = past_games_AvB["Target_Outcome_Loss"].sum() + won_by_B
            tied_AvB = past_games_AvB["Target_Outcome_Tie"].sum() +tied_AvB
            past_games_BvA =detailed_historical_matches_ranked[(detailed_historical_matches_ranked['home_team'] == teamB)&((detailed_historical_matches_ranked['away_team'] == teamA)) & (detailed_historical_matches_ranked["date"] < date) & (detailed_historical_matches_ranked['date'] > date- n_historic_window_f2f*pd.Timedelta(days=365) )].sort_values(by=['date'], ascending=False)
            #lastn = past_games.head(n_games)
            n_BvA = past_games_BvA.shape[0]
            goals_by_A = past_games_BvA["away_score"].sum() + goals_by_A
            goals_by_B = past_games_BvA["home_score"].sum() +goals_by_B
            won_by_A = past_games_BvA["Target_Outcome_Loss"].sum() +won_by_A
            won_by_B = past_games_BvA["Target_Outcome_Win"].sum() + won_by_B
            tied_AvB = past_games_BvA["Target_Outcome_Tie"].sum() +tied_AvB
            if (n_AvB + n_BvA)!=0:
                goals_by_A = goals_by_A/(n_AvB+n_BvA)
                goals_by_B = goals_by_B/(n_AvB+n_BvA)
            stats_val.append([goals_by_A,goals_by_B,won_by_A,won_by_B,tied_AvB])
        stats_cols = ["goals_AvB_past","goals_BvA_past","A_victories","B_victories","Team_ties"]
        stats_df = pd.DataFrame(stats_val, columns=stats_cols)
        detailed_historical_matches_ranked = pd.concat([detailed_historical_matches_ranked.reset_index(drop=True), stats_df], axis=1, ignore_index=False)
        detailed_historical_matches_ranked = detailed_historical_matches_ranked[detailed_historical_matches_ranked["date"]>="2000-01-01"] #Final Filter
        detailed_historical_matches_ranked_clean = detailed_historical_matches_ranked.fillna(0)
        detailed_historical_matches_ranked_clean = detailed_historical_matches_ranked_clean.reset_index(drop=True)
        detailed_historical_matches_ranked_clean.to_csv('data_loader/master_dataset_automated.csv',index=False)
        return 0
        
        
