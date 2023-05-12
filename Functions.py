from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
from keras.layers import Dropout
import numpy as np
from keras import regularizers

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def build_model(input_size_before,input_size_after):
    
    # Define the input layers
    inputs_timeseries = tf.keras.layers.Input(shape=(3, input_size_before+input_size_after+1))
    inputs_other = tf.keras.layers.Input(shape=(4,1))
    
    # Split the input into three separate tensors
    temp_input = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(inputs_timeseries)
    hum_input = tf.keras.layers.Lambda(lambda x: x[:, 1, :])(inputs_timeseries)
    co2_input = tf.keras.layers.Lambda(lambda x: x[:, 2, :])(inputs_timeseries)
    size_input = tf.keras.layers.Lambda(lambda x: x[:, 0])(inputs_other)
    room_input = tf.keras.layers.Lambda(lambda x: x[:, 1])(inputs_other)
    hour_input = tf.keras.layers.Lambda(lambda x: x[:, 2])(inputs_other)    
    day_input = tf.keras.layers.Lambda(lambda x: x[:, 3])(inputs_other)
    
    # Normalize the inputs
    temp_normalized = tf.keras.layers.BatchNormalization()(temp_input)
    hum_normalized = tf.keras.layers.BatchNormalization()(hum_input)
    co2_normalized = tf.keras.layers.BatchNormalization()(co2_input)
    size_normalized = tf.keras.layers.BatchNormalization()(size_input)
    room_normalized = tf.keras.layers.BatchNormalization()(room_input)
    hour_normalized = tf.keras.layers.Lambda(circular_normalization_hour)(hour_input)
    day_normalized = tf.keras.layers.Lambda(circular_normalization_day)(day_input)
    configuration= tf.keras.layers.Concatenate()([room_normalized, hour_normalized, day_normalized])
    configuration_normalized= tf.keras.layers.BatchNormalization()(configuration)
    
    # Define hidden layers for each input
    temp_hidden1 = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001))(temp_normalized)
    temp_dropout1 = Dropout(0.5)(temp_hidden1)
    temp_hidden2 = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001))(temp_dropout1)
    temp_dropout2 = Dropout(0.5)(temp_hidden2)
    temp_hidden3 = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001))(temp_dropout2)
    temp_dropout3 = Dropout(0.5)(temp_hidden3)
    hum_hidden1 = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001))(hum_normalized)
    hum_dropout1 = Dropout(0.5)(hum_hidden1)
    hum_hidden2 = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001))(hum_dropout1)
    hum_dropout2 = Dropout(0.5)(hum_hidden2)
    hum_hidden3 = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001))(hum_dropout2)
    hum_dropout3 = Dropout(0.5)(hum_hidden3)
    co2_hidden1 = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001))(co2_normalized)
    co2_dropout1 = Dropout(0.5)(co2_hidden1)
    co2_hidden2 = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001))(co2_dropout1)
    co2_dropout2 = Dropout(0.5)(co2_hidden2)
    co2_hidden3 = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001))(co2_dropout2)
    co2_dropout3 = Dropout(0.5)(co2_hidden3)
    configuration_hidden = tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.001))(configuration_normalized)
    configuration_hidden2 = tf.keras.layers.Dense(8, activation='relu',kernel_regularizer=regularizers.l2(0.001))(configuration_hidden)
    
    # Concatenate the inputs first line to use hour and day imput second line to not
    concat = tf.keras.layers.Concatenate()([temp_dropout3, hum_dropout3, co2_dropout3, size_normalized, configuration_hidden2])
    # concat = tf.keras.layers.Concatenate()([temp_dropout1, hum_dropout1, co2_dropout1, size_normalized])
    # concat =tf.keras.layers.Concatenate()([configuration_hidden2])
    concat_normalized = tf.keras.layers.BatchNormalization()(concat)
    # Define the hidden layers after concatenation
    hidden1 = tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001))(concat_normalized)
    dropout1 = Dropout(0.1)(hidden1)
    hidden2 = tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.001))(dropout1)
    dropout2 = Dropout(0.1)(hidden2)
    hidden3 = tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.001))(dropout2)
    dropout3 = Dropout(0.1)(hidden3)
    hidden4 = tf.keras.layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.001))(dropout3)
    dropout4 = Dropout(0.1)(hidden4)
    # Define the output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout4)

    # Define the model with separate inputs and hidden layers
    model = tf.keras.Model(inputs=[inputs_timeseries,inputs_other], outputs=output)

    
    #Training configuration
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, # Optimizer
        loss='binary_crossentropy', # Loss function to minimize
        metrics=['accuracy'], # List of metrics to monitor
    )

    return model

def circular_normalization_hour(x):
    sin = K.sin(x * (2.0 * np.pi / 24))
    cos = K.cos(x * (2.0 * np.pi / 24))
    return K.concatenate([sin, cos])

def circular_normalization_day(x):
    sin = K.sin(x * (2.0 * np.pi / 7))
    cos = K.cos(x * (2.0 * np.pi / 7))
    return K.concatenate([sin, cos])

def input_data_timeseries(input_size_before,input_size_after,t,dataset):
        
    data=[[],[],[]]
    for i in range(input_size_before+1):
        data[0].append(dataset['air_temperature'][t-i])
        data[1].append(dataset['relative_humidity'][t-i])
        data[2].append(dataset['co2_concentration'][t-i])
    for i in range(input_size_after):
        data[0].append(dataset['air_temperature'][t+i+1])
        data[1].append(dataset['relative_humidity'][t+i+1])
        data[2].append(dataset['co2_concentration'][t+i+1])   
    return data

def input_data_other(t,dataset):
    data=[]
    data.append(dataset['floor_area'][t])
    data.append(dataset['room_type'][t])
    data.append(dataset['Hour of the day'][t])
    data.append(dataset['Day number of the week'][t])
    return data

def output_data(t,dataset):
    data=[]
    data.append(dataset['occupancy_ground_truth'][t])
    return data
