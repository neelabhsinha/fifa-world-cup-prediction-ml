from data_loader.data import *
data = DataSetSelection()
new_data=DataSetGeneration()
output = data.features_binary("Canada","Cuba",date = '2022-11-11')
p = new_data.generate_masterdataset() #This generated a new file .
