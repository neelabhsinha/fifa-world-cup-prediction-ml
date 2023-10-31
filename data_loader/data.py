# This is the feature selector file .
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt
#from sklearn.feature_selection import SelectFromModel
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
#from sklearn.metrics import accuracy_score
#from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC
#from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")
#UNDER CONSTRUCTION
class DataSetSelection:
    def __init__(self):
        #Master dataset should be read here . It never changes. It is the csv with all features before truncation.
        #TODO : Paste all of the data reading code from the notebooks and test. Block - by - block ( Cell 10-13)
        # I am not sure if pandas dataset is passed
        #features_chosen_after_analysis = []
        #features_chosen_by_hand = []
        self.pca = False
        self.lda=False
        self.corr = False
        self.featurelist = []
        #self.daBigSet = pd.read_csv
        #TODO : Check this


    def features_binary(self,teamA,teamB,date,timeWindow = 2):
        """ UNDER CONSTRUCTION
          Returns team attributes for a match
        You may use this if you want . This returns all of the original features appended. Supervised learners may wanna use
        supervised_dataset_final for their learning set. I assume the tournament folks would need it during simulation.
        This might take a long time to run. Absolutely don't do it at every match node .
        args :
                teamA : string that is either the Full Name of the first team or the FIFA abbreviation
                teamB:  string that is either the Full Name of the second team or the FIFA abbreviation
                date: DateTime format "YYYY-MM-DD" when you intend to face them off .
                timeWindow : (Not implemented yet . How far do you want to compare their strengths)
        returns :
                [team_A_unary_attributes|team_B_unary_attributes|team_AvB_data] or PCAd sphagetti            
        """
        features_binary = []

        #TODO: Copy Notebook Code (Cell 17-20) in a sane manner
        #raise "NotImplementedError"
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
            masterdataset=pd.read_csv('data_loader/master_dataset.csv')
            masterdataset = masterdataset[(masterdataset["date"] >= date_start) & (masterdataset["date"] < date_end)].reset_index(drop=True) #Always filter by date first.
            featureset = [ "del_rank",'home_goals_mean','home_goals_mean_ln','home_goals_suf_mean','home_goals_suf_mean_ln', 'home_rank_mean',
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
            masterdataset=pd.read_csv("data_loader/CorrCorrectedData.csv")
            masterdataset["date"] = pd.to_datetime(masterdataset["date"]) 
            masterdataset = masterdataset[(masterdataset["date"] >= date_start) & (masterdataset["date"] < date_end)].reset_index(drop=True) #Always filter by date first.
            targetset = ["Target_Outcome_Win","Target_Outcome_Loss","Target_Outcome_Tie"]
            featureset = ["del_rank",'home_goals_mean','home_goals_mean_ln','home_goals_suf_mean', 
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

