from data_loader.data import *
data = DataSetSelection()
#Corrleation truncated Data
features,labels = data.supervised_dataset_final()
# Please pre-process like this.
# Labels are split like this so that you expliclty know what the targets are . Matched rowwise so no problem with permutations
# Join split,etc. for your training and validation flow after taking them to numpy arrays like this
features_np = features.to_numpy()


# In your training -testing flow the names of the team /dates shouldn't matter so excluded.
print(features.head(5))
print(features_np.shape)
#print(labels)
#untruncated Data
features,labels = data.supervised_dataset_final(dimred_method="None")
# Please pre-process like this.
# Labels are split like this so that you expliclty know what the targets are . Matched rowwise so no problem with permutations
# Join split,etc. for your training and validation flow after taking them to numpy arrays like this
features_np = features.to_numpy()


# In your training -testing flow the names of the team /dates shouldn't matter so excluded.
print(features.head(5))
print(features_np.shape)
#print(labels)
#PCA Data
features,labels = data.supervised_dataset_final(dimred_method="PCA")
# Please pre-process like this.
# Labels are split like this so that you expliclty know what the targets are . Matched rowwise so no problem with permutations
# Join split,etc. for your training and validation flow after taking them to numpy arrays like this
features_np = features.to_numpy()


# In your training -testing flow the names of the team /dates shouldn't matter so excluded.
print(features.head(5))
print(labels.head(5))
print(features_np.shape)

#Commented out because I checked an error
#features,labels = data.supervised_dataset_final(date_start='2008-01-01',date_end='2008-01-01')
# Please pre-process like this.
# Labels are split like this so that you expliclty know what the targets are . Matched rowwise so no problem with permutations
# Join split,etc. for your training and validation flow after taking them to numpy arrays like this
features,labels = data.supervised_dataset_final(date_start='2008-01-01',date_end='2018-01-01')
features_np = features.to_numpy()


# In your training -testing flow the names of the team /dates shouldn't matter so excluded.
print(features.head(5))
print(labels.head(5))
print(features_np.shape)

