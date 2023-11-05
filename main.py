# tune regularization for multinomial logistic regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import *
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from data_loader.data import *
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
# get the dataset
#def get_dataset():
#	data = DataSetSelection()
#    X,Y = data.supervised_dataset_final()
#    return
def get_dataset():
    data = DataSetSelection()
    standard_scaler = StandardScaler()
    X,Y =data.supervised_dataset_final(dimred_method="None",date_start="2000-01-01")
    z = Y[Y["Target_Outcome_Tie"] == 1]
    print(z.shape)
    z = Y[Y["Target_Outcome_Loss"] == 1]
    print(z.shape)
    z = Y[Y["Target_Outcome_Win"] == 1]
    print(z.shape)
    X= X.to_numpy()
    X = standard_scaler.fit_transform(X)
    Y["Cat"] = Y.apply(lambda row : 1 if (row.Target_Outcome_Win == 1) else -1 if (row.Target_Outcome_Loss==1) else 0,axis=1)
    #Y["Cat"] = Y.apply(lambda row : 1 if (row.Target_Outcome_Win == 1) else -1 ,axis=1)
    y = Y["Cat"]
    y = y.to_numpy()
    return X,y

# get a list of models to evaluate
def get_models():
    models = dict()
    for p in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
        # create name for model
        key = '%.4f' % p
        # turn off penalty in some cases
        if p == 0.0:
            # no penalty in this case
            models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='none',max_iter=10000)
        else:
            models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=p,max_iter=10000)
    return models

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    y_pred = cross_val_predict(model, X, y, cv=10)
    conf_mat = confusion_matrix(y, y_pred)
    #print("Confusion matrix below")
    #print(conf_mat)
    return scores
# generate master dataset (2 minutes)
#data_gen = DataSetGeneration()
#data_gen.generate_masterdataset()
# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
k= 1
for name, model in models.items():
    # evaluate the model and collect the scores
    scores = evaluate_model(model, X, y)
    # store the results
    results.append(scores)
    names.append(name)
    #print("scores are ",scores)
    # print("scores are ",scores)
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
