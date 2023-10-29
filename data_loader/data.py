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
        #Master dataset should be read here
        #TODO : Paste all of the data reading code from the notebooks and test. Block - by - block ( Cell 10-13)
        # I am not sure if pandas dataset is passed
        features_chosen_after_analysis = []
        features_chosen_by_hand = []
        #TODO : Check this


    def features_binary(self,teamA,teamB,date,timeWindow = 2):
        """ UNDER CONSTRUCTION
          Returns team attributes for a match
        You may use this if you want . This returns all of the original features appended. Supervised learners may wanna use
        supervised_dataset_final for their learning set. I assume the tournament folks would need it during simulation.
        
        args :
                teamA : string that is either the Full Name of the first team or the FIFA abbreviation
                teamB:  string that is either the Full Name of the second team or the FIFA abbreviation
                date: DateTime format "YYYY-MM-DD" when you intend to face them off .
                timeWindow : (Not implemented yet . How far do you want to compare their strengths)
        returns :
                [team_A_unary_attributes|team_B_unary_attributes|team_AvB_data]             
        """
        features_binary = []

        #TODO: Copy Notebook Code (Cell 17-20) in a sane manner
        raise "NotImplementedError"
        return features_binary
    def supervised_dataset_final(self,dimred_method = "None"):
        """ UNDER CONSTRUCTION
        This is the final reduced dataset we will work with.
        args:
            dimrec_medthod = Method for dimension reduction 
                                'PCA' for PCA
                                'LDA' for LDA (Not Implemented yet)
                                'PCA+LDA' PCA followed by LDA (Not Implemented yet.Don't count on it.)
                                'None' for only correlation reduced dataset
                                
        returns : [X_data|Y_data]                        """
        #TODO: Copy Notebook Code (Cell 17-20)
        raise "NotImplementedError"
        return dataset
    # Ignore unsupervised for now
    def unsupervised_features_final(self,pca="False"):
        """ UNDER CONSTRUCTION"""
        return 0

