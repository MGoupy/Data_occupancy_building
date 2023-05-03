import pandas as pd

from Functions import*

#%%
##############################################################################
### PARAMETERS ###
##############################################################################

param_input_size=20

#%%
##############################################################################
### LOAD DATA ###
##############################################################################

dataset = pd.read_csv('dataset_classification_hojbo.csv')

#%%
##############################################################################
### DATA CLEANING ###
##############################################################################

dataset_analyse=dataset.copy()

#%%
##############################################################################
### DATA TREATMENT ###
##############################################################################

#Split data for each appartment appartement 1 is the test appartment, appartments 2-4 are the training ones
dataset_test = dataset_analyse[(dataset_analyse['apt_no'] == 1)]

dataset_training_1 = dataset_analyse[(dataset_analyse['apt_no'] == 2)]
dataset_training_2 = dataset_analyse[(dataset_analyse['apt_no'] == 3)]
dataset_training_3 = dataset_analyse[(dataset_analyse['apt_no'] == 4)]
dataset_training_4 = dataset_analyse[(dataset_analyse['apt_no'] == 5)]

#%%
##############################################################################
### TRAINING / TEST DATA ###
##############################################################################

model = build_model(param_input_size)

print(input_data(20,50,dataset))
print(output_data(50,dataset))