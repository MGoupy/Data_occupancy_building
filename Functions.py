from keras import Sequential
import keras
import tensorflow as tf
import numpy as np


def build_model(input_size):
    
    #Model structure
    model=Sequential([
        #tf.keras.layers.Normalization(input_shape=[input_size,], axis=None),
        tf.keras.layers.Dense(100,input_shape=[input_size,], activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])
    
    #Training configuration
    model.compile(
        optimizer='adam', # Optimizer
        loss='binary_crossentropy', # Loss function to minimize
        metrics=['accuracy'], # List of metrics to monitor
    )

    return model

def input_data(input_size,t,dataset):
    data=[]
    for i in range(input_size):
        data.append(dataset['co2_concentration'][t-i])
    return np.array(data)

def output_data(t,dataset):
    data=[]
    data.append(dataset['occupancy_ground_truth'][t])
    data_1=[data]
    return np.array(np.array(data_1))