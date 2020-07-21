# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 11. 27
"""
import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras import layers
from keras.utils.generic_utils import get_custom_objects
from activation_functions import *

get_custom_objects().update({'swish': Activation(swish)})


def GRU_vol(input_dim, time_step, batch_size, epochs, activation, bias_initializer, kernel_initializer,
            bias_regularizer, hidden_layers, dropout, dropout_rate, batch_normalization, early_stop, x_train, y_train,
            lr):
    model = Sequential()
    model.add(layers.GRU(hidden_layers[0], input_shape=[time_step, input_dim],
                         activation=activation,
                         bias_initializer=bias_initializer,
                         kernel_initializer=kernel_initializer,
                         bias_regularizer=bias_regularizer,
                         return_sequences=True
                         ))
    if batch_normalization:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropout_rate))

    for hidden_layer in hidden_layers[1:]:
        model.add(layers.GRU(hidden_layer,
                             activation=activation,
                             bias_initializer=bias_initializer,
                             kernel_initializer=kernel_initializer,
                             return_sequences=True
                             ))
        if batch_normalization:
            model.add(BatchNormalization())
        if dropout:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.Adam(lr=lr))

    if early_stop:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  callbacks=[EarlyStopping(patience=10)],
                  validation_split=0.2)
    else:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0)

    return model
