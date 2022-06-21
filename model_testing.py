from numpy.core.defchararray import join
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


def validate(model, state, n=4096, comm=True, type=None):
    """Validate the performance of the model by generating random settings"""
    config.training_size = n
    dataset = generate_dataset(state, n)
    x, y_true = process_dataset(dataset)
    x_LHV = add_LHV(x)
    y_predict = tf.cast(model.predict(x_LHV), tf.float32)
    y_true = tf.cast(np.repeat(y_true, config.LHV_size, axis=0), tf.float32)
    if comm:
        score = comm_customLoss_multiple(y_true, y_predict, type=type)
    else:
        score = customLoss_multiple(y_true, y_predict, type=type)
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


def map_distr_TV(model, LHV_1, LHV_2, n=4096, same=False):
    """Generate the distribution for a vector pair model by fixing the LHVs.

    Args:
        model : the model
        LHV_1 (list): 3D unit vector for the first LHV
        LHV_2 (list): 3D unit vector for the second LHV
        n (int, optional): Number of random joint measurement settings. Defaults to 8192.

    Returns:
        dataframe: dataframe containing the distribution of the communication bit
    """
    if not same:
        vec_alice, vec_bob = random_joint_vectors(n)
    else:
        vec_alice = random_unit_vectors(n, 3)
        vec_bob = vec_alice
    LHVs = np.concatenate([LHV_1, LHV_2], axis=0).reshape(1, 6)
    config.number_of_LHV = 6
    input = np.concatenate(
        [vec_alice, vec_bob, np.repeat(LHVs, n, axis=0)], axis=1)
    output = model.predict(input)
    df = pd.DataFrame(np.concatenate((input, output), axis=1), columns=[
                      'ax', 'ay', 'az', 'bx', 'by', 'bz', 'L1x', 'L1y', 'L1z', 'L2x', 'L2y', 'L2z', 'c', 'p_1(a=+1)', 'p_1(a=-1)', 'p_1(b=+1)', 'p_1(b=-1)', 'p_2(a=+1)', 'p_2(a=-1)', 'p_2(b=+1)', 'p_2(b=-1)'])
    return df


def map_distr(model, LHV_1, n=4096, type="single vector"):
    """Generate the distribution for a single vector model by fixing the LHV vector.

    Args:
        model : the model
        LHV_1 (list): 3D unit vector for the first LHV
        n (int, optional): Number of random joint measurement settings. Defaults to 4096.
        type (str, optional): type of LHV to be used. Defaults to "single vector".

    Returns:
        dataframe: dataframe containing the distribution of the communication bit
    """
    vec_alice, vec_bob = random_joint_vectors(n)
    if type == "single vector":
        LHVs = np.concatenate([LHV_1, ], axis=0).reshape(1, 3)
        config.number_of_LHV = 3
        config.LHV_type = "single vector"
    elif type == "semicircle":
        LHVs = np.concatenate([LHV_1, ], axis=0).reshape(1, 2)
        config.number_of_LHV = 2
        config.LHV_type = "semicircle"
    input = np.concatenate(
        [vec_alice, vec_bob, np.repeat(LHVs, n, axis=0)], axis=1)
    output = model.predict(input)
    if type == "semicircle":
        df = pd.DataFrame(np.concatenate((input, output), axis=1), columns=[
            'ax', 'ay', 'az', 'bx', 'by', 'bz', 'L1x', 'L1z', 'c', 'p_1(a=+1)', 'p_1(a=-1)', 'p_1(b=+1)', 'p_1(b=-1)', 'p_2(a=+1)', 'p_2(a=-1)', 'p_2(b=+1)', 'p_2(b=-1)'])
    elif type == "single vector":
        df = pd.DataFrame(np.concatenate((input, output), axis=1), columns=[
            'ax', 'ay', 'az', 'bx', 'by', 'bz', 'L1x', 'L1y', 'L1z', 'c', 'p_1(a=+1)', 'p_1(a=-1)', 'p_1(b=+1)', 'p_1(b=-1)', 'p_2(a=+1)', 'p_2(a=-1)', 'p_2(b=+1)', 'p_2(b=-1)'])
    return df


def plot_comm_distr_vector(distr, type='spherical', color='comm', set_axes=None, savename=None, show=True, vmin=0, vmax=1, title=None):
    """Plot a comm distribution"""
    cdata = distr.c
    adata_1 = distr['p_1(a=+1)']
    adata_2 = distr['p_2(a=+1)']
    bdata_1 = distr['p_1(b=+1)']
    bdata_2 = distr['p_2(b=+1)']
    l1x, l1y, l1z = distr.L1x, distr.L1y, distr.L1z
    l2x, l2y, l2z = distr.L2x, distr.L2y, distr.L2z
    fig = None

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
        
    elif color == 'add_1':
        c = adata_1 + bdata_1
        axes = 'alice'
        vmin, vmax = 0.5, 1.5
    elif color == 'add_2':
        c = adata_2 + bdata_2
        axes = 'alice'
        vmin, vmax = 0.5, 1.5

    if set_axes:
        axes = set_axes

    if axes == 'alice':
        xdata, ydata, zdata = distr.ax, distr.ay, distr.az
    elif axes == 'bob':
        xdata, ydata, zdata = distr.bx, distr.by, distr.bz
    elif axes == 'lhv':
        xdata, ydata, zdata = distr.L1x, distr.L1y, distr.L1z

    if type == 'scatter':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        img = ax.scatter(ydata, xdata, zdata, c=c, vmin=vmin, vmax=vmax)
        ax.plot([0, l1y[0]*1.25], [0, l1x[0]*1.25], [0, l1z[0]*1.25], 'r-o', linewidth=5)
        ax.plot([0, l2y[0]*1.25], [0, l2x[0]*1.25], [0, l2z[0]*1.25], 'b-o', linewidth=5)
        
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_zlabel('z')
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        fig.colorbar(img, cax=cbar_ax)
        
        ax.set_ylim(ax.get_ylim()[::-1])
        
    elif type == 'spherical':
        theta_data = np.arccos(zdata)
        phi_data = np.arctan2(ydata, xdata)
        fig = plt.figure()
        ax = fig.add_subplot()
        img = ax.scatter(phi_data, theta_data, c=c, vmin=vmin, vmax=vmax)

        l1theta = np.arccos(l1z)
        l1phi = np.arctan2(l1y, l1x)
        l2theta = np.arccos(l2z)
        l2phi = np.arctan2(l2y, l2x)
        ax.scatter(l1phi, l1theta, color='red')
        ax.scatter(l2phi, l2theta, color='blue')

        ax.set_xlabel('phi', fontsize=22)
        ax.set_ylabel('theta', fontsize=22)
        if title:
            ax.set_title(title, fontsize=26)
        fig.subplots_adjust(right=0.8)
        fig.subplots_adjust(bottom=0.15)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(img, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=18)

    if savename:
        plt.savefig(savename)
    if show:
        plt.show()
    return fig


def map_distr_SV_party(model, vec_alice=[0, 0, 1], vec_bob=[0, 0, 1], n=4096):
    """Generate the distribution for a vector model by fixing the input vectors.

    Args:
        model : the model
        vec_alice (list): 3D unit vector for Alice's input
        vec_bob (list): 3D unit vector for Bob's input
        n (int, optional): Number of LHVs. Defaults to 4096.

    Returns:
        dataframe: dataframe containing the distribution of the communication bit
    """
    config.number_of_LHV = 3
    config.LHV_size = n
    xdata = add_LHV(np.array([vec_alice+vec_bob, ]))
    output = model.predict(xdata)
    df = pd.DataFrame(np.concatenate((xdata, output), axis=1), columns=[
                      'ax', 'ay', 'az', 'bx', 'by', 'bz', 'L1x', 'L1y', 'L1z', 'c', 'p_1(a=+1)', 'p_1(a=-1)', 'p_1(b=+1)', 'p_1(b=-1)', 'p_2(a=+1)', 'p_2(a=-1)', 'p_2(b=+1)', 'p_2(b=-1)'])
    return df


def evaluate_marginals(model, theta, vec_alice, vec_bob, singlet=True, local=False, strategy=0):
    config.training_size = 1
    config.LHV_size = 5000
    xdata = np.array([np.concatenate([vec_alice, vec_bob]), ])
    if not local:
        output = predict(model, xdata)[0]
    else:
        output = predict_local(model, xdata, strategy=strategy)[0]
    if singlet:
        print('Marginal of Alice')
        print('Predicted :', output[0]+output[1]-output[2]-output[3])
        print('Theory    :', np.cos(2 * theta) * vec_alice[2])
        print('Marginal of Bob')
        print('Predicted :', output[0]-output[1]+output[2]-output[3])
        print('Theory    :', -np.cos(2 * theta) * vec_bob[2])
        print('Joint Marginal')
        print('Predicted :', output[0]-output[1]-output[2]+output[3])
        print('Theory    :', -vec_alice[2] * vec_bob[2] - np.sin(2 * theta)
              * (vec_alice[0] * vec_bob[0] + vec_alice[1] * vec_bob[1]))
    else:
        print('Marginal of Alice')
        print('Predicted :', output[0]+output[1]-output[2]-output[3])
        print('Theory    :', np.cos(2 * theta) * vec_alice[2])
        print('Marginal of Bob')
        print('Predicted :', output[0]-output[1]+output[2]-output[3])
        print('Theory    :', np.cos(2 * theta) * vec_bob[2])
        print('Joint Marginal')
        print('Predicted :', output[0]-output[1]-output[2]+output[3])
        print('Theory    :', vec_alice[2] * vec_bob[2] + np.sin(2 * theta)
              * (vec_alice[0] * vec_bob[0] - vec_alice[1] * vec_bob[1]))


def plot_marginal_alice_semicircle(model, m=100, n=10000, fix=False):
    """Generate the distribution for a single vector model by fixing the LHV vector.

    Args:
        model : the model
        n (int, optional): Number of random joint measurement settings.

    Returns:
        dataframe: dataframe containing the distribution of the communication bit
    """
    vec_alice, vec_bob = random_joint_vectors(n)
    if fix:
        vec_bob = np.repeat([[0, np.sqrt(1/2), np.sqrt(1/2)]], n, axis=0)
    for i in range(m):
        LHV = random_semicircle_vector()
        LHVs = np.concatenate([LHV, ], axis=0).reshape(1, 2)
        input = np.concatenate(
            [vec_alice, vec_bob, np.repeat(LHVs, n, axis=0)], axis=1)
        output = model.predict(input)
        output_df = pd.DataFrame(output, columns=[
                                 'c', 'p_1(a=+1)', 'p_1(a=-1)', 'p_1(b=+1)', 'p_1(b=-1)', 'p_2(a=+1)', 'p_2(a=-1)', 'p_2(b=+1)', 'p_2(b=-1)'])
        if i == 0:
            alice_df = (
                output_df.c * output_df['p_1(a=+1)'] + (1 - output_df.c) * output_df['p_2(a=+1)'])
            bob_df = (
                output_df.c * output_df['p_1(b=+1)'] + (1 - output_df.c) * output_df['p_2(b=+1)'])
            joint_df = (2 * alice_df * bob_df - alice_df - bob_df + 1)
        else:
            alice_df = (output_df.c * output_df['p_1(a=+1)'] + (
                1 - output_df.c) * output_df['p_2(a=+1)']) + alice_df
            bob_df = (output_df.c * output_df['p_1(b=+1)'] +
                      (1 - output_df.c) * output_df['p_2(b=+1)']) + bob_df
            a_df = (output_df.c * output_df['p_1(a=+1)'] +
                    (1 - output_df.c) * output_df['p_2(a=+1)'])
            b_df = (output_df.c * output_df['p_1(b=+1)'] +
                    (1 - output_df.c) * output_df['p_2(b=+1)'])
            joint_df = (2 * a_df * b_df - a_df - b_df + 1) + joint_df
    alice_df = 1/m * alice_df
    bob_df = 1/m * bob_df
    joint_df = 1/m * joint_df
    df = pd.DataFrame(np.concatenate(
        [vec_alice, vec_bob, alice_df.to_frame(), bob_df.to_frame(), joint_df.to_frame()], axis=1), columns=[
        'ax', 'ay', 'az', 'bx', 'by', 'bz', 'p(a=+1)', 'p(b=+1)', 'p(ab=+1)'])

    xdata, ydata, zdata = df.ax, df.ay, df.az
    theta_data = np.arccos(zdata)
    phi_data = np.arctan2(ydata, xdata)
    fig = plt.figure()
    ax = fig.add_subplot()
    img = ax.scatter(phi_data, theta_data, c=2 *
                     df['p(a=+1)']-1, vmin=-1, vmax=1)
    ax.set_xlabel('phi')
    ax.set_ylabel('theta')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    fig.colorbar(img, cax=cbar_ax)
    plt.show()

    xdata, ydata, zdata = df.bx, df.by, df.bz
    theta_data = np.arccos(zdata)
    phi_data = np.arctan2(ydata, xdata)
    fig = plt.figure()
    ax = fig.add_subplot()
    img = ax.scatter(phi_data, theta_data, c=2 *
                     df['p(b=+1)']-1, vmin=-1, vmax=1)
    ax.set_xlabel('phi')
    ax.set_ylabel('theta')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    fig.colorbar(img, cax=cbar_ax)
    plt.show()

    xdata, ydata, zdata = df.ax, df.ay, df.az
    theta_data = np.arccos(zdata)
    phi_data = np.arctan2(ydata, xdata)
    fig = plt.figure()
    ax = fig.add_subplot()
    img = ax.scatter(phi_data, theta_data, c=2 *
                     df['p(ab=+1)']-1, vmin=-1, vmax=1)
    ax.set_xlabel('phi')
    ax.set_ylabel('theta')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    fig.colorbar(img, cax=cbar_ax)
    plt.show()

    return df


def predict_local(model, x, strategy=0):
    """Predict the probability distributions given a model and some inputs x"""
    x_LHV = add_LHV(x)
    y_predict = model.predict(x_LHV)
    probs_predict = comm_customLoss_local_distr_multiple(y_predict)[:,4*strategy:4*strategy+4]
    return probs_predict


def comm_balance(model, vec_alice, vec_bob):
    """Predict the probability distributions given a model and some inputs x"""
    config.LHV_size = 100000
    x = np.array([np.concatenate([vec_alice, vec_bob]), ])
    x_LHV = add_LHV(x)
    y_predict = model.predict(x_LHV)
    comms = comm_customLoss_local_distr_multiple(y_predict)[:,-1]
    return np.average(comms)