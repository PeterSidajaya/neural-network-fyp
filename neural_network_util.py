import config
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, Dot

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
    group_LHV = Lambda(lambda x: x[:, 6:], output_shape=(
        (number_of_LHV,)))(inputTensor)

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
    number_of_LHV = config.number_of_LHV  # Number of hidden variables, i.e. 6 for vector pair
    depth = config.party_depth
    width = config.party_width
    outputsize = config.party_outputsize
    activ = config.activation_func
    activ2 = 'softmax'
    activ3 = config.activation_func_comm
    # 6 numbers (two 3D vectors) plus one hidden variable as inputs.
    inputTensor = Input((6+number_of_LHV,))

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:, 0:3], output_shape=((3,)))(inputTensor)
    group_beta = Lambda(lambda x: x[:, 3:6], output_shape=((3,)))(inputTensor)
    group_LHV = Lambda(lambda x: x[:, 6:6+number_of_LHV],
                       output_shape=((number_of_LHV,)))(inputTensor)

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

    # Output the two probability distributions and combine them according to the formula
    """ This is WRONG! We can't weigh before the outer product as it means more than one bit of
    information is transferred to Bob.
    outputTensor1 = Concatenate()([group_a1, group_b1])
    outputTensor2 = Concatenate()([group_a2, group_b2])
    outputTensor = Lambda(
        lambda x: x[0] * x[1] + (1.0 - x[0]) * x[2])([group_c, outputTensor1, outputTensor2])
    """
    outputTensor = Concatenate()(
        [group_c, group_a1, group_b1, group_a2, group_b2])

    model = Model(inputTensor, outputTensor)
    return model


def build_NewModel():
    """Build a model with one bit of communication between parties.
    """
    if not (config.LHV_type == "vector pair"):
        raise ValueError("LHV must use vector pair!")
    # Number of hidden variables, i.e. 6 for vector pair
    number_of_LHV = config.number_of_LHV
    depth = config.party_depth
    width = config.party_width
    outputsize = config.party_outputsize
    activ = config.activation_func
    activ2 = 'softmax'
    activ3 = config.activation_func_comm
    # 6 numbers (two 3D vectors) plus one hidden variable as inputs.
    inputTensor = Input((6 + 6,))

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:, 0:3], output_shape=((3,)))(inputTensor)
    group_beta = Lambda(lambda x: x[:, 3:6], output_shape=((3,)))(inputTensor)

    group_LHV_1 = Lambda(lambda x: x[:, 6:9], output_shape=((3,)))(inputTensor)
    group_alpha_dot_1 = Dot(axes=1)([group_alpha, group_LHV_1])
    group_beta_dot_1 = Dot(axes=1)([group_beta, group_LHV_1])

    group_LHV_2 = Lambda(lambda x: x[:, 9:12],
                         output_shape=((3,)))(inputTensor)
    group_alpha_dot_2 = Dot(axes=1)([group_alpha, group_LHV_2])
    group_beta_dot_2 = Dot(axes=1)([group_beta, group_LHV_2])

    group_lhv_dot = Dot(axes=1)([group_LHV_1, group_LHV_2])

    """
    Used in some prototypes and carry-over models. Basically the same.
    group_LHV = Lambda(lambda x: x[:, 6:12], output_shape=((6,)))(inputTensor)
    group_alpha_dot_1 = Lambda(lambda x: x[:,0:1]*x[:,6:7] + x[:,1:2]*x[:,7:8] + x[:,2:3]*x[:,8:9], output_shape=((1,)))(inputTensor)
    group_alpha_dot_2 = Lambda(lambda x: x[:,0:1]*x[:,9:10] + x[:,1:2]*x[:,10:11] + x[:,2:3]*x[:,11:12], output_shape=((1,)))(inputTensor)
    group_beta_dot_1 = Lambda(lambda x: x[:,3:4]*x[:,6:7] + x[:,4:5]*x[:,7:8] + x[:,5:6]*x[:,8:9], output_shape=((1,)))(inputTensor)
    group_beta_dot_2 = Lambda(lambda x: x[:,3:4]*x[:,9:10] + x[:,4:5]*x[:,10:11] + x[:,5:6]*x[:,11:12], output_shape=((1,)))(inputTensor)
    """

    # Route hidden variables to parties Alice and Bob
    group_a = Concatenate()(
        [group_alpha, group_LHV_1, group_LHV_2, group_alpha_dot_1, group_alpha_dot_2, group_lhv_dot])
    group_b = Concatenate()(
        [group_beta, group_LHV_1, group_LHV_2, group_beta_dot_1, group_beta_dot_2, group_lhv_dot])
    group_c = Concatenate()(
        [group_alpha, group_LHV_1, group_LHV_2, group_alpha_dot_1, group_alpha_dot_2, group_lhv_dot])

    # Neural network at the parties Alice, Bob
    # Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    kernel_init = tf.keras.initializers.VarianceScaling(
        scale=2, mode='fan_in', distribution='truncated_normal', seed=None)
    for _ in range(depth):
        group_a1 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_a)
        group_b1 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_b)
        group_a2 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_a)
        group_b2 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_b)
        group_c = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_c)

    # Apply final softmax layer
    group_a1 = Dense(outputsize, activation=activ2)(group_a1)
    group_b1 = Dense(outputsize, activation=activ2)(group_b1)
    group_a2 = Dense(outputsize, activation=activ2)(group_a2)
    group_b2 = Dense(outputsize, activation=activ2)(group_b2)
    group_c = Dense(1, activation=activ3)(group_c)

    outputTensor = Concatenate()(
        [group_c, group_a1, group_b1, group_a2, group_b2])

    model = Model(inputTensor, outputTensor)
    return model


def build_NewModel_NC():
    """Build a no-communication model of simulation.
    """
    number_of_LHV = config.number_of_LHV  # Number of hidden variables, i.e. alpha, beta, gamma
    depth = config.party_depth
    width = config.party_width
    outputsize = config.party_outputsize
    activ = config.activation_func
    activ2 = 'softmax'
    # 6 numbers (two 3D vectors) plus one hidden variable as inputs.
    inputTensor = Input((6+6,))

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:, 0:3], output_shape=((3,)))(inputTensor)
    group_beta = Lambda(lambda x: x[:, 3:6], output_shape=((3,)))(inputTensor)

    group_LHV_1 = Lambda(lambda x: x[:, 6:9], output_shape=((3,)))(inputTensor)
    group_alpha_dot_1 = Dot(axes=1)([group_alpha, group_LHV_1])
    group_beta_dot_1 = Dot(axes=1)([group_beta, group_LHV_1])

    group_LHV_2 = Lambda(lambda x: x[:, 9:12],
                         output_shape=((3,)))(inputTensor)
    group_alpha_dot_2 = Dot(axes=1)([group_alpha, group_LHV_2])
    group_beta_dot_2 = Dot(axes=1)([group_beta, group_LHV_2])

    # Route hidden variables to visibile parties Alice and Bob
    group_a = Concatenate()(
        [group_alpha, group_LHV_1, group_LHV_2, group_alpha_dot_1, group_alpha_dot_2])
    group_b = Concatenate()(
        [group_beta, group_LHV_1, group_LHV_2, group_beta_dot_1, group_beta_dot_2])

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


def build_NewModel_Afix():
    """Build a model with one bit of communication between parties.
    """
    if not (config.LHV_type == "vector pair"):
        raise ValueError("LHV must use vector pair!")
    # Number of hidden variables, i.e. 6 for vector pair
    number_of_LHV = config.number_of_LHV
    depth = config.party_depth
    width = config.party_width
    outputsize = config.party_outputsize
    activ = config.activation_func
    activ2 = 'softmax'
    activ3 = config.activation_func_comm
    # 6 numbers (two 3D vectors) plus one hidden variable as inputs.
    inputTensor = Input((6 + 6,))

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:, 0:3], output_shape=((3,)))(inputTensor)
    group_beta = Lambda(lambda x: x[:, 3:6], output_shape=((3,)))(inputTensor)

    group_LHV_1 = Lambda(lambda x: x[:, 6:9], output_shape=((3,)))(inputTensor)
    group_alpha_dot_1 = Dot(axes=1)([group_alpha, group_LHV_1])
    group_beta_dot_1 = Dot(axes=1)([group_beta, group_LHV_1])

    group_LHV_2 = Lambda(lambda x: x[:, 9:12],
                         output_shape=((3,)))(inputTensor)
    group_alpha_dot_2 = Dot(axes=1)([group_alpha, group_LHV_2])
    group_beta_dot_2 = Dot(axes=1)([group_beta, group_LHV_2])

    group_lhv_dot = Dot(axes=1)([group_LHV_1, group_LHV_2])

    # Route hidden variables to parties Alice and Bob
    group_a = Concatenate()(
        [group_alpha, group_LHV_1, group_LHV_2, group_alpha_dot_1, group_alpha_dot_2, group_lhv_dot])
    group_b = Concatenate()(
        [group_beta, group_LHV_1, group_LHV_2, group_beta_dot_1, group_beta_dot_2, group_lhv_dot])
    group_c = Concatenate()(
        [group_alpha, group_LHV_1, group_LHV_2, group_alpha_dot_1, group_alpha_dot_2, group_lhv_dot])

    # Neural network at the parties Alice, Bob
    # Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    kernel_init = tf.keras.initializers.VarianceScaling(
        scale=2, mode='fan_in', distribution='truncated_normal', seed=None)
    for _ in range(depth):
        group_a = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_a)
        group_b1 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_b)
        group_b2 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_b)
        group_c = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_c)

    # Apply final softmax layer
    group_a = Dense(outputsize, activation=activ2)(group_a)
    group_b1 = Dense(outputsize, activation=activ2)(group_b1)
    group_b2 = Dense(outputsize, activation=activ2)(group_b2)
    group_c = Dense(1, activation=activ3)(group_c)

    outputTensor = Concatenate()(
        [group_c, group_a, group_b1, group_a, group_b2])

    model = Model(inputTensor, outputTensor)
    return model


def build_Model_v3():
    """Build a model with one bit of communication between parties.
    """
    if not (config.LHV_type == "vector pair"):
        raise ValueError("LHV must use vector pair!")
    # Number of hidden variables, i.e. 6 for vector pair
    number_of_LHV = config.number_of_LHV
    depth = config.party_depth
    width = config.party_width
    outputsize = config.party_outputsize
    activ = config.activation_func
    activ2 = 'softmax'
    activ3 = config.activation_func_comm
    # 6 numbers (two 3D vectors) plus one hidden variable as inputs.
    inputTensor = Input((6 + 6,))

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:, 0:3], output_shape=((3,)))(inputTensor)
    group_beta = Lambda(lambda x: x[:, 3:6], output_shape=((3,)))(inputTensor)

    group_LHV_1 = Lambda(lambda x: x[:, 6:9], output_shape=((3,)))(inputTensor)
    group_alpha_dot_1 = Dot(axes=1)([group_alpha, group_LHV_1])
    group_beta_dot_1 = Dot(axes=1)([group_beta, group_LHV_1])

    group_LHV_2 = Lambda(lambda x: x[:, 9:12],
                         output_shape=((3,)))(inputTensor)
    group_alpha_dot_2 = Dot(axes=1)([group_alpha, group_LHV_2])
    group_beta_dot_2 = Dot(axes=1)([group_beta, group_LHV_2])

    group_lhv_dot = Dot(axes=1)([group_LHV_1, group_LHV_2])

    # Route hidden variables to parties Alice and Bob
    group_a = Concatenate()(
        [group_alpha, group_LHV_1, group_LHV_2, group_alpha_dot_1, group_alpha_dot_2, group_lhv_dot])
    group_b = Concatenate()(
        [group_beta, group_LHV_1, group_LHV_2, group_beta_dot_1, group_beta_dot_2, group_lhv_dot])
    group_c = Concatenate()(
        [group_alpha, group_LHV_1, group_LHV_2, group_alpha_dot_1, group_alpha_dot_2, group_lhv_dot])

    # Neural network at the parties Alice, Bob
    # Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    kernel_init = tf.keras.initializers.VarianceScaling(
        scale=2, mode='fan_in', distribution='truncated_normal', seed=None)
    
    group_a1 = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_a)
    group_b1 = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_b)
    group_a2 = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_a)
    group_b2 = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_b)
    group_c = Dense(width, activation=activ,
                    kernel_initializer=kernel_init)(group_c)

    for _ in range(depth-1):
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

    outputTensor = Concatenate()(
        [group_c, group_a1, group_b1, group_a2, group_b2])

    model = Model(inputTensor, outputTensor)
    return model


def build_Model_qutrit():
    """Build a model with one bit of communication between parties.
    """
    # Number of hidden variables, i.e. 6 for vector pair
    number_of_LHV = config.number_of_LHV
    depth = config.party_depth
    width = config.party_width
    outputsize = config.party_outputsize
    activ = config.activation_func
    activ2 = 'softmax'
    activ3 = config.activation_func_comm
    # 6 numbers (two 3D vectors) plus one hidden variable as inputs.
    inputTensor = Input((6 + config.number_of_LHV,))

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:, 0:3], output_shape=((3,)))(inputTensor)
    group_beta = Lambda(lambda x: x[:, 3:6], output_shape=((3,)))(inputTensor)
    group_LHV = Lambda(lambda x: x[:, 6:9], output_shape=((3,)))(inputTensor)

    # Route hidden variables to parties Alice and Bob
    group_a = Concatenate()(
        [group_alpha, group_LHV])
    group_b = Concatenate()(
        [group_beta, group_LHV])
    group_c = Concatenate()(
        [group_alpha, group_LHV])

    # Neural network at the parties Alice, Bob
    # Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    kernel_init = tf.keras.initializers.VarianceScaling(
        scale=2, mode='fan_in', distribution='truncated_normal', seed=None)
    
    group_a1 = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_a)
    group_b1 = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_b)
    group_a2 = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_a)
    group_b2 = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_b)
    group_c = Dense(width, activation=activ,
                    kernel_initializer=kernel_init)(group_c)

    for _ in range(depth-1):
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

    outputTensor = Concatenate()(
        [group_c, group_a1, group_b1, group_a2, group_b2])

    model = Model(inputTensor, outputTensor)
    return model

def build_Model_qutrit_NC():
    """Build a model with one bit of communication between parties.
    """
    # Number of hidden variables, i.e. 6 for vector pair
    number_of_LHV = config.number_of_LHV
    depth = config.party_depth
    width = config.party_width
    outputsize = config.party_outputsize
    activ = config.activation_func
    activ2 = 'softmax'
    activ3 = config.activation_func_comm
    # 6 numbers (two 3D vectors) plus one hidden variable as inputs.
    inputTensor = Input((6 + config.number_of_LHV,))

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:, 0:3], output_shape=((3,)))(inputTensor)
    group_beta = Lambda(lambda x: x[:, 3:6], output_shape=((3,)))(inputTensor)
    group_LHV = Lambda(lambda x: x[:, 6:9], output_shape=((3,)))(inputTensor)

    # Route hidden variables to parties Alice and Bob
    group_a = Concatenate()(
        [group_alpha, group_LHV])
    group_b = Concatenate()(
        [group_beta, group_LHV])

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

    outputTensor = Concatenate()(
        [group_a, group_b])

    model = Model(inputTensor, outputTensor)
    return model


def keras_distance(p, q, type=None):
    """ Distance used in loss function."""
    if not type:
        p = K.clip(p, K.epsilon(), 1)
        q = K.clip(q, K.epsilon(), 1)
        return K.sum(p * K.log(p / q), axis=-1)
    elif type == 'tvd':
        return K.sum(K.abs(p - q))/2


def customLoss_distr(y_pred):
    """ Converts the output of the neural network to a probability vector.
    That is from a shape of (batch_size, outputsize + outputsize) to a shape of (outputsize * outputsize,)
    DEPRECIATED
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
    """ Custom loss function.
    DEPRECIATED
    """
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


def customLoss_multiple(y_true, y_pred, type=None):
    """ Custom loss function."""
    # Note that y_true is just LHV_size copies of the target distributions. So any row could be taken here. We just take 0-th row.
    probs_list = customLoss_distr_multiple(y_pred)
    loss = 0
    for i in range(config.training_size):
        loss += keras_distance(y_true[config.LHV_size*i, :], probs_list[i], type=type)
    return loss / config.training_size


def comm_customLoss_distr_multiple(y_pred):
    """ Converts the output of the neural network to several probability vectors.
    That is from a shape of (batch_size, 1 + outputsize + outputsize + outputsize + outputsize)
    to a shape of (batch_size, outputsize * outputsize).
    Used for the communication model only!
    """
    outputsize = config.party_outputsize
    LHV_size = config.LHV_size
    probs_list = []
    for i in range(config.training_size):
        c_probs = y_pred[LHV_size*i:LHV_size*(i+1), 0:1]
        a_probs_1 = y_pred[LHV_size*i:LHV_size*(i+1), 1:1*outputsize+1]
        b_probs_1 = y_pred[LHV_size*i:LHV_size *
                           (i+1), 1*outputsize+1:2*outputsize+1]
        a_probs_2 = y_pred[LHV_size*i:LHV_size *
                           (i+1), 2*outputsize+1:3*outputsize+1]
        b_probs_2 = y_pred[LHV_size*i:LHV_size *
                           (i+1), 3*outputsize+1:4*outputsize+1]

        a_probs_1 = K.reshape(a_probs_1, (-1, outputsize, 1))
        b_probs_1 = K.reshape(b_probs_1, (-1, 1, outputsize))

        a_probs_2 = K.reshape(a_probs_2, (-1, outputsize, 1))
        b_probs_2 = K.reshape(b_probs_2, (-1, 1, outputsize))

        probs_1 = a_probs_1 * b_probs_1
        probs_1 = K.reshape(probs_1, (-1, outputsize * outputsize))
        probs_2 = a_probs_2 * b_probs_2
        probs_2 = K.reshape(probs_2, (-1, outputsize * outputsize))
        probs = Lambda(lambda x: x[0] * x[1] + (1.0 - x[0])
                       * x[2])([c_probs, probs_1, probs_2])
        probs = K.mean(probs, axis=0)

        probs_list.append(probs)
    return probs_list


def comm_customLoss_multiple(y_true, y_pred, type=None):
    """ Custom loss function. Used for the communication model only!"""
    # Note that y_true is just LHV_size copies of the target distributions. So any row could be taken here. We just take 0-th row.
    probs_list = comm_customLoss_distr_multiple(y_pred)
    loss = 0
    for i in range(config.training_size):
        loss += keras_distance(y_true[config.LHV_size*i, :], probs_list[i], type=type)
    return loss / config.training_size


def CGLMP_local():
    """Build a no-communication model of simulation.
    """
    config.number_of_LHV  = 1   # Number of hidden variables, i.e. alpha, beta, gamma
    depth = config.party_depth
    width = config.party_width
    outputsize = config.party_outputsize
    activ = config.activation_func
    activ2 = 'softmax'
    # 6 numbers (two 3D vectors) plus one hidden variable as inputs.
    inputTensor = Input((3,))

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:, 0:1], output_shape=((1,)))(inputTensor)
    group_beta = Lambda(lambda x: x[:, 1:2], output_shape=((1,)))(inputTensor)

    group_LHV = Lambda(lambda x: x[:, 2:3], output_shape=((1,)))(inputTensor)

    # Route hidden variables to visibile parties Alice and Bob
    group_a = Concatenate()(
        [group_alpha, group_LHV])
    group_b = Concatenate()(
        [group_beta, group_LHV])

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

def CGLMP_nonlocal():
    """Build a model with one bit of communication between parties.
    """
    config.number_of_LHV  = 1   # Number of hidden variables, i.e. alpha, beta, gamma
    depth = config.party_depth
    width = config.party_width
    outputsize = config.party_outputsize
    activ = config.activation_func
    activ2 = 'softmax'
    activ3 = config.activation_func_comm
    # 6 numbers (two 3D vectors) plus one hidden variable as inputs.
    inputTensor = Input((3,))

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:, 0:1], output_shape=((1,)))(inputTensor)
    group_beta = Lambda(lambda x: x[:, 1:2], output_shape=((1,)))(inputTensor)

    group_LHV = Lambda(lambda x: x[:, 2:3], output_shape=((1,)))(inputTensor)

    # Route hidden variables to parties Alice and Bob
    group_a = Concatenate()(
        [group_alpha, group_LHV])
    group_b = Concatenate()(
        [group_beta, group_LHV])
    group_c = Concatenate()(
        [group_alpha, group_LHV])

    # Neural network at the parties Alice, Bob
    # Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    kernel_init = tf.keras.initializers.VarianceScaling(
        scale=2, mode='fan_in', distribution='truncated_normal', seed=None)
    for _ in range(depth):
        group_a1 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_a)
        group_b1 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_b)
        group_a2 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_a)
        group_b2 = Dense(width, activation=activ,
                         kernel_initializer=kernel_init)(group_b)
        group_c = Dense(width, activation=activ,
                        kernel_initializer=kernel_init)(group_c)

    # Apply final softmax layer
    group_a1 = Dense(outputsize, activation=activ2)(group_a1)
    group_b1 = Dense(outputsize, activation=activ2)(group_b1)
    group_a2 = Dense(outputsize, activation=activ2)(group_a2)
    group_b2 = Dense(outputsize, activation=activ2)(group_b2)
    group_c = Dense(1, activation=activ3)(group_c)

    outputTensor = Concatenate()(
        [group_c, group_a1, group_b1, group_a2, group_b2])

    model = Model(inputTensor, outputTensor)
    return model

def comm_customLoss_local_distr_multiple(y_pred):
    """ Converts the output of the neural network to several probability vectors.
    That is from a shape of (batch_size, 1 + outputsize + outputsize + outputsize + outputsize)
    to a shape of (batch_size, 2 * outputsize * outputsize + 1).
    Used for the communication model only!
    """
    outputsize = config.party_outputsize
    LHV_size = config.LHV_size
    probs_list = []
    for i in range(config.training_size):
        c_probs = y_pred[LHV_size*i:LHV_size*(i+1), 0:1]
        a_probs_1 = y_pred[LHV_size*i:LHV_size*(i+1), 1:1*outputsize+1]
        b_probs_1 = y_pred[LHV_size*i:LHV_size *
                           (i+1), 1*outputsize+1:2*outputsize+1]
        a_probs_2 = y_pred[LHV_size*i:LHV_size *
                           (i+1), 2*outputsize+1:3*outputsize+1]
        b_probs_2 = y_pred[LHV_size*i:LHV_size *
                           (i+1), 3*outputsize+1:4*outputsize+1]

        a_probs_1 = K.reshape(a_probs_1, (-1, outputsize, 1))
        b_probs_1 = K.reshape(b_probs_1, (-1, 1, outputsize))

        a_probs_2 = K.reshape(a_probs_2, (-1, outputsize, 1))
        b_probs_2 = K.reshape(b_probs_2, (-1, 1, outputsize))

        probs_1 = a_probs_1 * b_probs_1
        probs_1 = K.reshape(probs_1, (-1, outputsize * outputsize))
        probs_2 = a_probs_2 * b_probs_2
        probs_2 = K.reshape(probs_2, (-1, outputsize * outputsize))
        probs = Concatenate()([probs_1, probs_2, c_probs])
        probs = K.mean(probs, axis=0)

        probs_list.append(probs)
    return np.array(probs_list)

