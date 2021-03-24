from colour import Color
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qutip as qt

def plot_measurements(vector_alice_list, vector_bob_list):
    num = length(vector_alice_list)
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
    data = pd.read_csv(filename, index_col=0, names=['parameter', 'loss'])
    sns.set_theme()
    g = sns.relplot(x=0, y=1, kind="line", data=data)
    g.set_xlabels(xlabel)
    g.set_ylabels(ylabel)
    plt.title(title)
    plt.show()

