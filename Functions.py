from keras import Sequential
from keras import backend as K
import tensorflow as tf
from keras import regularizers
from focal_loss import BinaryFocalLoss
import numpy as np

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
    temp_hidden = tf.keras.layers.Dense(64, activation='relu')(temp_normalized)
    temp_hidden1 = tf.keras.layers.Dense(64, activation='relu')(temp_hidden)
    temp_hidden2 = tf.keras.layers.Dense(64, activation='relu')(temp_hidden1)
    hum_hidden = tf.keras.layers.Dense(64, activation='relu')(hum_normalized)
    hum_hidden1 = tf.keras.layers.Dense(64, activation='relu')(hum_hidden)
    hum_hidden2 = tf.keras.layers.Dense(64, activation='relu')(hum_hidden1)
    co2_hidden = tf.keras.layers.Dense(64, activation='relu')(co2_normalized)
    co2_hidden1 = tf.keras.layers.Dense(64, activation='relu')(co2_hidden)
    co2_hidden2 = tf.keras.layers.Dense(64, activation='relu')(co2_hidden1)
    configuration_hidden = tf.keras.layers.Dense(16, activation='relu')(configuration_normalized)
    configuration_hidden2 = tf.keras.layers.Dense(2, activation='relu')(configuration_hidden)
    
    # Concatenate the inputs first line to use hour and day imput second line to not
    concat = tf.keras.layers.Concatenate()([temp_hidden2, hum_hidden2, co2_hidden2, size_normalized, configuration_hidden2])
    # concat = tf.keras.layers.Concatenate()([temp_hidden, hum_hidden, co2_hidden, size_normalized])
    
    # Define the hidden layers after concatenation
    hidden1 = tf.keras.layers.Dense(64, activation='relu')(concat)
    hidden2 = tf.keras.layers.Dense(32, activation='relu')(hidden1)
    hidden3 = tf.keras.layers.Dense(32, activation='relu')(hidden2)
    hidden4 = tf.keras.layers.Dense(32, activation='relu')(hidden3)

    # Define the output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden4)

    # Define the model with separate inputs and hidden layers
    model = tf.keras.Model(inputs=[inputs_timeseries,inputs_other], outputs=output)

    
    #Training configuration
    model.compile(
        optimizer='adam', # Optimizer
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