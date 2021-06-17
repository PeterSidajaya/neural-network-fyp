from neural_network_util import *
from preprocess import *
from distribution_generator import *
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import config
import pandas as pd
import matplotlib.pyplot as plt


def predict_dataset(model, x):
    x_LHV = add_LHV(x)
    y_predict = model.predict(x_LHV)
    probs_predict = comm_customLoss_distr_multiple(y_predict)
    return probs_predict


def comm_distr(model, n=2048):
    vec_alice, vec_bob = random_joint_vectors(n)
    LHVs = np.linspace(0, 1, num=11)
    input = np.concatenate([np.tile(vec_alice, (11, 1)), np.tile(vec_bob, (11, 1)), np.repeat(
        LHVs.reshape(11, 1), n, axis=0)], axis=1)
    output = model.predict(input)
    df = pd.DataFrame(np.concatenate((input, output), axis=1), columns=[
                      'ax', 'ay', 'az', 'bx', 'by', 'bz', 'LHV', 'c', 'p_1(a=0)', 'p_1(a=1)', 'p_1(b=0)', 'p_1(b=1)', 'p_2(a=0)', 'p_2(a=1)', 'p_2(b=0)', 'p_2(b=1)'])
    return df


def plot_comm_distr_single(distr, index, axis, type='spherical'):
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


def plot_comm_distr(distr, type='spherical'):
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


def validate(model, state, n=4096):
    config.training_size = n
    dataset = generate_dataset(state, n)
    x, y_true = process_dataset(dataset)
    x_LHV = add_LHV(x)
    y_predict = tf.cast(model.predict(x_LHV), tf.float32)
    y_true = tf.cast(np.repeat(y_true, config.LHV_size, axis=0), tf.float32)
    score = comm_customLoss_multiple(y_true, y_predict)
    return score


def evaluate(model, filename):
    x, y_true = open_dataset(filename)
    config.training_size = len(y_true)
    x_LHV = add_LHV(x)
    y_predict = tf.cast(model.predict(x_LHV), tf.float32)
    y_true = tf.cast(np.repeat(y_true, config.LHV_size, axis=0), tf.float32)
    score = comm_customLoss_multiple(y_true, y_predict)
    return score


def comm_distr_TV(model, LHV_1, LHV_2, n=8192):
    vec_alice, vec_bob = random_joint_vectors(n)
    LHVs = np.concatenate([LHV_1, LHV_2], axis=0).reshape(1, 6)
    input = np.concatenate(
        [vec_alice, vec_bob, np.repeat(LHVs, n, axis=0)], axis=1)
    output = model.predict(input)
    df = pd.DataFrame(np.concatenate((input, output), axis=1), columns=[
                      'ax', 'ay', 'az', 'bx', 'by', 'bz', 'L1x', 'L1y', 'L1z', 'L2x', 'L2y', 'L2z', 'c', 'p_1(a=0)', 'p_1(a=1)', 'p_1(b=0)', 'p_1(b=1)', 'p_2(a=0)', 'p_2(a=1)', 'p_2(b=0)', 'p_2(b=1)'])
    return df


def plot_comm_distr_TV(distr, type='spherical'):
    xdata = distr.ax
    ydata = distr.ay
    zdata = distr.az
    cdata = distr.c
    if type == 'scatter3D':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xdata, ydata, zdata, c=cdata)
    elif type == 'spherical':
        theta_data = np.arccos(zdata)
        phi_data = np.arctan2(ydata, xdata)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        img = ax.scatter(phi_data, theta_data, c=cdata, vmin=0, vmax=1)
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
        fig.colorbar(img, cax=cbar_ax)
    plt.show()