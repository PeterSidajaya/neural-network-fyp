from colour import Color
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qutip as qt
import numpy as np


def plot_measurements(vector_alice_list, color=True):
    num = len(vector_alice_list)
    b = qt.Bloch()
    if color:
        b.vector_color = list(map(lambda x: x.rgb, list(
            Color('red').range_to(Color('purple'), num))))
    for i in range(num):
        vector_alice = vector_alice_list[i]
        b.add_vectors(vector_alice)
    b.show()
    plt.show()
    

def read_vectors(filename):
    df = pd.read_csv(filename, index_col=0)
    df = df.to_numpy()
    vector_list = df[::4, 0:6]
    num_of_vector = int(np.sqrt(len(vector_list[:, 0])))
    alice_vectors = vector_list[::num_of_vector, 0:3]
    bob_vectors = vector_list[:num_of_vector, 3:6]
    return alice_vectors, bob_vectors


def plot_dataset(filename, xlabel='X axis', ylabel='Y axis', title='Title'):
    data = pd.read_csv(filename, index_col=0, header=0, names=[
                       'parameter', 'loss']).sort_values('parameter')
    sns.set_theme()
    sns.set_context("paper")
    plt.figure(figsize=(6, 6))
    plt.plot(data.parameter, data.loss, 'o', color='red')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.show()
