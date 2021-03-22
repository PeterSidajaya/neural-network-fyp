import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import config

"""This file contains all the functions needed to create the Neural Network.
"""


def build_model():
    """Build a no-communication model of simulation.
    """
    number_of_LHV = config.number_of_LHV  # Number of hidden variables, i.e. alpha, beta, gamma
    depth = config.party_depth
    width = config.party_width
    outputsize = config.party_outputsize
    activ = config.activation_func
    activ2 = 'softmax'
    # 6 numbers (two 3D vectors) plus one hidden variable as inputs.
    inputTensor = Input((6+number_of_LHV,))

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:, 0:3], output_shape=((3,)))(inputTensor)
    group_beta = Lambda(lambda x: x[:, 3:6], output_shape=((3,)))(inputTensor)
    group_LHV = Lambda(lambda x: x[:, 6:7], output_shape=((1,)))(inputTensor)

    # Route hidden variables to visibile parties Alice and Bob
    group_a = Concatenate()([group_alpha, group_LHV])
    group_b = Concatenate()([group_beta, group_LHV])

    # Neural network at the parties Alice, Bob
    # Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    kernel_init = tf.keras.initializers.VarianceScaling(
        scale=2, mode='fan_in', distribution='truncated_normal', seed=None)
    for _ in range(depth):
        group_a = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_a)
        group_b = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_b)

    # Apply final softmax layer
    group_a = Dense(outputsize, activation=activ2)(group_a)
    group_b = Dense(outputsize, activation=activ2)(group_b)

    outputTensor = Concatenate()([group_a, group_b])

    model = Model(inputTensor, outputTensor)
    return model


def build_model_comm():
    """Build a model with one bit of communication between parties.
    """
    number_of_LHV = config.number_of_LHV  # Number of hidden variables, i.e. alpha, beta, gamma
    depth = config.party_depth
    width = config.party_width
    outputsize = config.party_outputsize
    activ = config.activation_func
    activ2 = 'softmax'
    activ3 = 'sigmoid'
    # 6 numbers (two 3D vectors) plus one hidden variable as inputs.
    inputTensor = Input((6+number_of_LHV,))

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:, 0:3], output_shape=((3,)))(inputTensor)
    group_beta = Lambda(lambda x: x[:, 3:6], output_shape=((3,)))(inputTensor)
    group_LHV = Lambda(lambda x: x[:, 6:7], output_shape=((1,)))(inputTensor)

    # Route hidden variables to visibile parties Alice and Bob
    group_a1 = Concatenate()([group_alpha, group_LHV])
    group_b1 = Concatenate()([group_beta, group_LHV])
    group_a2 = Concatenate()([group_alpha, group_LHV])
    group_b2 = Concatenate()([group_beta, group_LHV])
    group_c = Concatenate()([group_alpha, group_LHV])

    # Neural network at the parties Alice, Bob
    # Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    kernel_init = tf.keras.initializers.VarianceScaling(
        scale=2, mode='fan_in', distribution='truncated_normal', seed=None)
    for _ in range(depth):
        group_a1 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_a1)
        group_b1 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_b1)
        group_a2 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_a2)
        group_b2 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_b2)
        group_c = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_c)

    # Apply final softmax layer
    group_a1 = Dense(outputsize, activation=activ2)(group_a1)
    group_b1 = Dense(outputsize, activation=activ2)(group_b1)
    group_a2 = Dense(outputsize, activation=activ2)(group_a2)
    group_b2 = Dense(outputsize, activation=activ2)(group_b2)
    group_c = Dense(1, activation=activ3)(group_c)

    outputTensor1 = Concatenate()([group_a1, group_b1])
    outputTensor2 = Concatenate()([group_a2, group_b2])
    outputTensor = Lambda(
        lambda x: x[0] * x[1] + (1.0 - x[0]) * x[2])([group_c, outputTensor1, outputTensor2])

    model = Model(inputTensor, outputTensor)
    return model


def keras_distance(p, q):
    """ Distance used in loss function. """
    p = K.clip(p, K.epsilon(), 1)
    q = K.clip(q, K.epsilon(), 1)
    return K.sum(p * K.log(p / q), axis=-1)


def customLoss_distr(y_pred):
    """ Converts the output of the neural network to a probability vector.
    That is from a shape of (batch_size, outputsize + outputsize) to a shape of (outputsize * outputsize,)
    """
    outputsize = config.party_outputsize
    a_probs = y_pred[:, 0:outputsize]
    b_probs = y_pred[:, outputsize: outputsize + outputsize]

    # Do an outer product
    a_probs = K.reshape(a_probs, (-1, outputsize, 1))
    b_probs = K.reshape(b_probs, (-1, 1, outputsize))

    probs = a_probs*b_probs
    probs = K.mean(probs, axis=0)
    probs = K.flatten(probs)
    return probs


def customLoss(y_true, y_pred):
    """ Custom loss function."""
    # Note that y_true is just LHV_size copies of the target distributions. So any row could be taken here. We just take 0-th row.
    return keras_distance(y_true[0, :], customLoss_distr(y_pred))


def customLoss_distr_multiple(y_pred):
    """ Converts the output of the neural network to several probability vectors.
    That is from a shape of (batch_size, outputsize + outputsize) to a shape of (training_size, outputsize * outputsize)
    """
    outputsize = config.party_outputsize
    LHV_size = config.LHV_size
    probs_list = []
    for i in range(config.training_size):
        a_probs = y_pred[LHV_size*i:LHV_size*(i+1), 0:outputsize]
        b_probs = y_pred[LHV_size*i:LHV_size *
                         (i+1), outputsize: outputsize + outputsize]

        a_probs = K.reshape(a_probs, (-1, outputsize, 1))
        b_probs = K.reshape(b_probs, (-1, 1, outputsize))

        probs = a_probs*b_probs
        probs = K.mean(probs, axis=0)
        probs = K.flatten(probs)
        probs_list.append(probs)
    return probs_list


def customLoss_multiple(y_true, y_pred):
    """ Custom loss function."""
    # Note that y_true is just LHV_size copies of the target distributions. So any row could be taken here. We just take 0-th row.
    probs_list = customLoss_distr_multiple(y_pred)
    loss = 0
    for i in range(config.training_size):
        loss += keras_distance(y_true[config.LHV_size*i, :], probs_list[i])
    return loss / config.training_size
