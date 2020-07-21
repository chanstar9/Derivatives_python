# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 11. 21
"""
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from keras.utils.generic_utils import get_custom_objects
from activation_functions import *

get_custom_objects().update({'swish': Activation(swish)})


def DNN_vol(input_dim, batch_size, epochs, activation, bias_initializer, kernel_initializer, x_train, y_train,
            hidden_layers, lr, bias_regularizer, dropout, dropout_rate, batch_normalization, early_stop):
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_dim=input_dim,
                    activation=activation,
                    bias_initializer=bias_initializer,
                    kernel_initializer=kernel_initializer,
                    bias_regularizer=bias_regularizer
                    ))
    if batch_normalization:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(dropout_rate))

    for hidden_layer in hidden_layers[1:]:
        model.add(Dense(hidden_layer,
                        activation=activation,
                        bias_initializer=bias_initializer,
                        kernel_initializer=kernel_initializer
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
