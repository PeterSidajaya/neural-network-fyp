import numpy as np
import math
from numpy.random import default_rng
import matplotlib.pyplot as plt
from numpy import ndarray
from evo_protocols import *
rng = default_rng()


def random_vector(n):
    """Generate an uniformly distributed random vector."""
    components = [rng.standard_normal() for i in range(n)]
    r = math.sqrt(sum(x*x for x in components))
    v = np.array([x/r for x in components])
    return v


def random_vectors(m, n):
    array = np.random.normal(size=(m, n))
    norm = np.linalg.norm(array, axis=1)
    return array/norm[:, None]


def random_joint_vectors(n):
    """Generate a list of random 3D unit vectors."""
    a, b = [], []
    for _ in range(n):
        vector_a = random_vector(3)
        vector_b = random_vector(3)
        a.append(vector_a)
        b.append(vector_b)
    return (a, b)


def generate_protocol(protocol, lhv):
    data = np.ndarray((3, 10000))
    for i in range(10000):
        vector = random_vector(3)
        data[0, i], data[1, i] = spherical(vector)
        res = protocol(vector, lhv)
        if isinstance(res, ndarray):
            data[2, i] = res[0]
        else:
            data[2, i] = res
    return data


def plot_protocol(protocol, lhv, optional=None, savename=None, show=True, title=None):
    data = generate_protocol(protocol, lhv)
    plt.scatter(data[0], data[1], c=data[2], vmin=0, vmax=1)

    phi, theta = spherical(lhv[0])
    plt.scatter(phi, theta, c='red')

    phi, theta = spherical(lhv[1])
    plt.scatter(phi, theta, c='blue')
    
    if isinstance(optional, type(np.ndarray(1))):
        phi, theta = spherical(optional)
        plt.scatter(phi, theta, c='green')

    if title:
        plt.title(title)

    if savename:
        plt.savefig(savename)
    
    if show:
        plt.show()


def spherical(vector):
    return (np.arctan2(vector[1], vector[0]), np.arccos(vector[2]))
