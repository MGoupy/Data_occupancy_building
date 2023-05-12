import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from numpy import *
from Functions import *
from random import randrange
from keras.callbacks import EarlyStopping
from sklearn.feature_selection import RFE

#%%
##############################################################################
### PARAMETERS ###
##############################################################################

param_input_before=36
param_input_after=36
param_add_occupancy=1
param_add_void=1

#%%
##############################################################################
### LOAD DATA ###
##############################################################################

dataset = pd.read_csv('dataset_classification_hojbo.csv')
"""
#check ow much more time the romm is empty than occupied in the training sample 
room_occ=0
room_empty=0
for i in range(len(dataset['train_test_data'])):
    if dataset['train_test_data'][i] == 'training_data':
        if dataset['occupancy_ground_truth'][i]==0:
            room_empty+=1
        else:
            room_occ+=1
print(room_occ)
print(room_empty)
"""


#%%
##############################################################################
### DATA CLEANING ###
##############################################################################

dataset_analyse=dataset.copy()

#%%
##############################################################################
### DATA TREATMENT ###
##############################################################################

#Create training data for the model
data_training = [[],[],[]] #training data for x and y
room1=0
room2=0
for apt in [2,3,4,5]:
    df=dataset_analyse[(dataset_analyse['apt_no'] == apt)]
    df = df.reset_index(drop=True)
    for i in range(len(df['apt_no'])-param_input_before-param_input_after):
        if df['room_no'][i] == df['room_no'][i+param_input_after+param_input_before]:
            if output_data(i+param_input_before,df)[0]==1:
                add=param_add_occupancy
            else:
                add=param_add_void
            for j in range(add):
                #k=randrange(len(data_training[0])+1) #To randomize the inputs
                k=len(data_training[0])
                data_training[0].insert(k,input_data_timeseries(param_input_before,param_input_after,i+param_input_before,df))
                data_training[1].insert(k,input_data_other(i+param_input_before,df))
                data_training[2].insert(k,output_data(i+param_input_before,df))

          
#Create test data data for the model
data_test=[[],[],[]]
df=dataset_analyse[(dataset_analyse['apt_no'] == 1)]
#df=df[(dataset_analyse['room_no'] == 1)]
df = df.reset_index(drop=True)
for i in range(len(df['apt_no'])-param_input_before-param_input_after):
    if df['room_no'][i] == df['room_no'][i+param_input_before]:
        data_test[0].append(input_data_timeseries(param_input_before,param_input_after,i+param_input_before,df))
        data_test[1].append(input_data_other(i+param_input_before,df))
        data_test[2].append(output_data(i+param_input_before,df))

#%%
##############################################################################
### MODEL CREATION ###
##############################################################################

model = build_model(param_input_before,param_input_after)
#%%
##############################################################################
### TRAINING ###
##############################################################################

early_stop = EarlyStopping(monitor='val_loss', patience=500)
history = model.fit(
    [tf.convert_to_tensor(data_training[0]),tf.convert_to_tensor(data_training[1])],
    tf.convert_to_tensor(data_training[2]),
    batch_size=64,
    epochs=500,
    validation_data=([tf.convert_to_tensor(data_test[0]),tf.convert_to_tensor(data_test[1])],tf.convert_to_tensor(data_test[2])),
    callbacks=[early_stop]
)

#%%
##############################################################################
### TESTING ###
##############################################################################


input_test = [np.array(data_test[0]),np.array(data_test[1])]
prediction=model.predict(input_test)

y_true=[]
for i in range (len(data_test[2])):
    y_true.append(data_test[2][i][0])
y_pred=[]
for i in range (len(prediction)):
    if prediction[i]>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


#%%
##############################################################################
### METRICS ###
##############################################################################

"Confusion matrix"
# create the confusion matrix based on 
cm = confusion_matrix(y_true, y_pred)
print(cm)

tn, fp, fn, tp = cm.ravel()

print("True positives: ", tp)
print("True negatives: ", tn)
print("False positives: ", fp)
print("False negatives: ", fn)

"Percision and Recall"
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)

#percision and recall balance above 0.8
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

#%%
##############################################################################
### PLOTS ###
##############################################################################


fig1 = plt.figure()
y_pred_plot=y_pred[:]
y_true_plot=y_true[:]
y_prob_plot=prediction[:]
plt.plot(y_pred_plot,'r')
plt.plot(y_true_plot,'g--')
plt.plot(y_prob_plot,'b')

fig2 = plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

fig3 = plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()