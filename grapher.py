from colour import Color
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qutip as qt


def plot_measurements(vector_alice_list, vector_bob_list):
    num = len(vector_alice_list)
    b = qt.Bloch()
    b.vector_color = list(map(lambda x: x.rgb, list(
        Color('red').range_to(Color('purple'), num))))
    for i in range(num):
        vector_alice = vector_alice_list[i]
        vector_bob = vector_bob_list[i]
        b.add_vectors(vector_alice)
        b.add_vectors(vector_bob)
    b.show()
    plt.show()


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


plot_dataset('werner_result_with_comm.csv', xlabel='Werner parameter',
             ylabel='Relative entropy', title='Werner state loss')
