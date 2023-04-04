import keras
import tensorflow as tf
from keras import Input, Model
from keras.layers import (
    Dense, Reshape, Flatten, LeakyReLU,
    LayerNormalization, Dropout, BatchNormalization
    )

def build_generator(latent_space, n_var, n_features=2,use_bias=False,activation1 = 'relu',activation2='tanh'):

    model = tf.keras.Sequential()
    model.add(Dense(n_var*3, input_shape=(latent_space,), activation=activation1, use_bias=use_bias))
    model.add(LayerNormalization())
    model.add(Dense(n_var*4, activation=activation1))
    model.add(LayerNormalization())
    model.add(Dense((n_var*4), activation=activation1))
    model.add(LayerNormalization())
    model.add(Dense(n_features, activation=activation2, use_bias=use_bias))

    return model

def build_critic(n_var, use_bias=False, n_features=2):
    n_var = n_var*2
    model = tf.keras.Sequential()
    model.add(Dense(n_var*3, input_shape=(n_features,), use_bias=use_bias))
    model.add(LayerNormalization())
    model.add(LeakyReLU())
    model.add(Dense(n_var*5))
    model.add(LeakyReLU())
    model.add(Dense(n_var*5))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1))

    return model


