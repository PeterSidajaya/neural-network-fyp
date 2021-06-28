from neural_network_util import *
from preprocess import *
from distribution_generator import *
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import config
import pandas as pd
import matplotlib.pyplot as plt


def predict(model, x):
    """Predict the probability distributions given a model and some inputs x"""
    x_LHV = add_LHV(x)
    y_predict = model.predict(x_LHV)
    probs_predict = comm_customLoss_distr_multiple(y_predict)
    return probs_predict


def map_distr(model, n=2048):
    """Create a dataframe of the distribution of the communication bit sent in a model
    Only for single number LHV"""
    vec_alice, vec_bob = random_joint_vectors(n)
    LHVs = np.linspace(0, 1, num=11)
    input = np.concatenate([np.tile(vec_alice, (11, 1)), np.tile(vec_bob, (11, 1)), np.repeat(
        LHVs.reshape(11, 1), n, axis=0)], axis=1)
    output = model.predict(input)
    df = pd.DataFrame(np.concatenate((input, output), axis=1), columns=[
                      'ax', 'ay', 'az', 'bx', 'by', 'bz', 'LHV', 'c', 'p_1(a=0)', 'p_1(a=1)', 'p_1(b=0)', 'p_1(b=1)', 'p_2(a=0)', 'p_2(a=1)', 'p_2(b=0)', 'p_2(b=1)'])
    return df


def plot_comm_distr_single(distr, index, axis, type='spherical'):
    """Plot the communication bit sent using the comm distribution
    Helper function for the below function, only for single number LHV"""
    LHVs = np.linspace(0, 1, num=11)
    distr_sliced = distr[distr['LHV'] == LHVs[index]]
    xdata = distr_sliced.ax
    ydata = distr_sliced.ay
    zdata = distr_sliced.az
    cdata = distr_sliced.c
    if type == 'spherical':
        theta_data = np.arccos(zdata)
        phi_data = np.arctan2(ydata, xdata)
        img = axis.scatter(phi_data, theta_data, c=cdata, vmin=0, vmax=1)
    return img


def plot_comm_distr_number(distr, type='spherical'):
    """Plot the communication bit sent using the comm distribution for multiple values of LHV
    Only for single number LHV"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    fig.delaxes(axes[2, 3])

    for i in range(3):
        for j in range(4):
            if (i == 2 and j == 3):
                break
            axes[i, j].set_aspect(1)
            axes[i, j].set_xlabel('phi')
            axes[i, j].set_ylabel('theta')
            axes[i, j].set_title(
                'lambda = ' + str(round((i * 4 + j) * 0.1, 2)))
            img = plot_comm_distr_single(
                distr, i * 4 + j, axes[i, j], type=type)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
    fig.colorbar(img, cax=cbar_ax)
    plt.show()


def validate(model, state, n=4096, comm=True):
    """Validate the performance of the model by generating random settings"""
    config.training_size = n
    dataset = generate_dataset(state, n)
    x, y_true = process_dataset(dataset)
    x_LHV = add_LHV(x)
    y_predict = tf.cast(model.predict(x_LHV), tf.float32)
    y_true = tf.cast(np.repeat(y_true, config.LHV_size, axis=0), tf.float32)
    if comm:
        score = comm_customLoss_multiple(y_true, y_predict)
    else:
        score = customLoss_multiple(y_true, y_predict)
    return score


def evaluate(model, filename):
    """Validate the performance of the model using a dataset"""
    x, y_true = open_dataset(filename)
    config.training_size = len(y_true)
    x_LHV = add_LHV(x)
    y_predict = tf.cast(model.predict(x_LHV), tf.float32)
    y_true = tf.cast(np.repeat(y_true, config.LHV_size, axis=0), tf.float32)
    score = comm_customLoss_multiple(y_true, y_predict)
    return score


def map_distr_TV(model, LHV_1, LHV_2, n=4096):
    """Generate the distribution for a vector pair model.

    Args:
        model : the model
        LHV_1 (list): 3D unit vector for the first LHV
        LHV_2 (list): 3D unit vector for the second LHV
        n (int, optional): Number of random joint measurement settings. Defaults to 8192.

    Returns:
        dataframe: dataframe containing the distribution of the communication bit
    """
    vec_alice, vec_bob = random_joint_vectors(n)
    LHVs = np.concatenate([LHV_1, LHV_2], axis=0).reshape(1, 6)
    config.number_of_LHV = 6
    input = np.concatenate(
        [vec_alice, vec_bob, np.repeat(LHVs, n, axis=0)], axis=1)
    output = model.predict(input)
    df = pd.DataFrame(np.concatenate((input, output), axis=1), columns=[
                      'ax', 'ay', 'az', 'bx', 'by', 'bz', 'L1x', 'L1y', 'L1z', 'L2x', 'L2y', 'L2z', 'c', 'p_1(a=0)', 'p_1(a=1)', 'p_1(b=0)', 'p_1(b=1)', 'p_2(a=0)', 'p_2(a=1)', 'p_2(b=0)', 'p_2(b=1)'])
    return df


def map_distr_SV(model, LHV_1, n=4096):
    """Generate the distribution for a vector model.

    Args:
        model : the model
        LHV_1 (list): 3D unit vector for the first LHV
        LHV_2 (list): 3D unit vector for the second LHV
        n (int, optional): Number of random joint measurement settings. Defaults to 8192.

    Returns:
        dataframe: dataframe containing the distribution of the communication bit
    """
    vec_alice, vec_bob = random_joint_vectors(n)
    LHVs = np.concatenate([LHV_1,], axis=0).reshape(1, 3)
    config.number_of_LHV = 3
    input = np.concatenate(
        [vec_alice, vec_bob, np.repeat(LHVs, n, axis=0)], axis=1)
    output = model.predict(input)
    df = pd.DataFrame(np.concatenate((input, output), axis=1), columns=[
                      'ax', 'ay', 'az', 'bx', 'by', 'bz', 'L1x', 'L1y', 'L1z', 'c', 'p_1(a=0)', 'p_1(a=1)', 'p_1(b=0)', 'p_1(b=1)', 'p_2(a=0)', 'p_2(a=1)', 'p_2(b=0)', 'p_2(b=1)'])
    return df


def plot_comm_distr_vector(distr, type='spherical', color='comm', savename=None, show=True):
    """Plot a comm distribution for a vector pair model"""
    cdata = distr.c
    adata_1 = distr['p_1(a=0)']
    adata_2 = distr['p_2(a=0)']
    bdata_1 = distr['p_1(b=0)']
    bdata_2 = distr['p_2(b=0)']
    
    if color == 'comm':
        c = cdata
        axes = 'alice'
    elif color == 'alice_1':
        c = adata_1
        axes = 'alice'
    elif color == 'alice_2':
        c = adata_2
        axes = 'alice'
    elif color == 'bob_1':
        c = bdata_1
        axes = 'bob'
    elif color == 'bob_2':
        c = bdata_2
        axes = 'bob'
    
    if axes == 'alice':
        xdata = distr.ax
        ydata = distr.ay
        zdata = distr.az
    elif axes == 'bob':
        xdata = distr.bx
        ydata = distr.by
        zdata = distr.bz
    
    if type == 'scatter':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        img = ax.scatter(xdata, ydata, zdata, c=c, vmin=0, vmax=1)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        fig.colorbar(img, cax=cbar_ax)
    elif type == 'spherical':
        theta_data = np.arccos(zdata)
        phi_data = np.arctan2(ydata, xdata)
        fig = plt.figure()
        ax = fig.add_subplot()
        img = ax.scatter(phi_data, theta_data, c=c, vmin=0, vmax=1)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        fig.colorbar(img, cax=cbar_ax)
        
    if savename:
        plt.savefig(savename)
    if show:
        plt.show()
