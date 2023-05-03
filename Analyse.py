import pandas as pd
from numpy import *
from Functions import*

#%%
##############################################################################
### PARAMETERS ###
##############################################################################

param_input_size=20
param_train_max=5000

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

data_training_x=[]
for i in range(len(dataset_analyse['apt_no'])-param_input_size):
    data_training_x.append(input_data(param_input_size,i+param_input_size,dataset_analyse))
    
data_training_y=[]
for i in range(len(dataset_analyse['apt_no'])-param_input_size):
    data_training_y.append(output_data(i+param_input_size,dataset_analyse))


#%%
##############################################################################
### MODEL CREATION ###
##############################################################################

model = build_model(param_input_size)



#%%
##############################################################################
### TRAINING ###
##############################################################################

history = model.fit(
    #data_training_x,
    #data_training_y,
    #expand_dims(data_training_x,axis=-1),
    #expand_dims(data_training_y,axis=-1),
    np.array(data_training_x),
    np.array(data_training_y),
    batch_size=32,
    epochs=5
)
#%%
##############################################################################
### TESTING ###
##############################################################################

input_test = data_training_x[:5]
input_test = tf.reshape(input_test,shape=(5,20))
#input_test = np.array(input_test)
prediction=model.predict(input_test)

print("input:")
print(input_test)
print("output:")
print(prediction)