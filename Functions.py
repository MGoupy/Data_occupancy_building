from keras import Sequential
import tensorflow as tf

def build_model(input_size):
    
    single_feature_normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis=None)
    
    model=Sequential([
        #tf.keras.layers.Input(shape=(input_size,)),
        single_feature_normalizer,
        tf.keras.layers.Dense(input_size),
        tf.keras.layers.Dense(1)
    ])
    return model

def input_data(input_size,t,dataset):
    data=[]
    for i in range(input_size):
        data.append(dataset['co2_concentration'][t-i])
    return data

def output_data(t,dataset):
    return dataset['occupancy_ground_truth'][t]