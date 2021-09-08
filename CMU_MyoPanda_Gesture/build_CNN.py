#### PACKAGE IMPORTS ####
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
import pandas as pd


def get_CNN_model(input_shape, output_shape, wd, dr = 0.3):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), 
                     input_shape = (input_shape[1], input_shape[2], 1), activation = 'relu', 
                     padding = 'same', name = 'conv_1_input', kernel_regularizer = tf.keras.regularizers.l2(wd)))
    model.add(Dropout(dr, name = 'Dropout_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'Conv_2', kernel_regularizer = tf.keras.regularizers.l2(wd)))
    model.add(Dropout(dr, name = 'Dropout_2'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same', name = 'Conv_3', kernel_regularizer = tf.keras.regularizers.l2(wd)))
    model.add(Dropout(dr, name = 'Dropout_3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), name = 'Maxpool_1'))
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu', name = 'Dense_1', kernel_regularizer = tf.keras.regularizers.l2(wd)))
    model.add(Dropout(dr, name = 'Dropout_4'))
    model.add(Dense(output_shape, activation = 'softmax', name = 'out_layer'))
    
    return model