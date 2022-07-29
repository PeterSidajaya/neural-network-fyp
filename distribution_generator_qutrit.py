import qutip as qt
import numpy as np
import pandas as pd
import math

"""This file contains the functions needed to create the datasets."""


S_x, S_y, S_z = qt.jmat(1)

def random_unit_vector(n):
    """Generate a random unit vector of n dimensions."""
    components = [np.random.normal() for i in range(n)]
    r = math.sqrt(sum(x*x for x in components))
    v = [x/r for x in components]
    return v


def random_unit_vectors(m, n):
    """Generate m random vectors of n dimensions."""
    array = np.random.normal(size=(m, n))
    norm = np.linalg.norm(array, axis=1)
    return array / norm[:, None]


def operator_dot(vector):
    """Calculate the dot product of a vector with the Pauli matrices vector."""
    return vector[0] * qt.sigmax() + vector[1] * qt.sigmay() + vector[2] * qt.sigmaz()


def spin_operator(vector):
    S = vector[0] * S_x + vector[1] * S_y + vector[2] * S_z
    eigvals, eigstates = S.eigenstates()
    if max(eigvals - [-1,0,1]) > 1e-10:
        print('ERROR!')
        raise(ValueError)
    return {-1:eigstates[0], 0:eigstates[1], 1:eigstates[2]}


def probability_dict(state, vector_a, vector_b):
    """Returns the probabilities of a joint measurement defined by two unit vectors on an entangled state."""
    prob = {}
    spin_a = spin_operator(vector_a)
    spin_b = spin_operator(vector_b)
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            op_a = qt.ket2dm(spin_a[i])
            op_b = qt.ket2dm(spin_b[j])
            prob[i, j] = (qt.tensor(op_a, op_b)
                                          * state).tr()
    return prob


def probability_list(state, vector_a, vector_b):
    """Returns the probabilities of a joint measurement defined by two unit vectors on an entangled state."""
    prob = []
    spin_a = spin_operator(vector_a)
    spin_b = spin_operator(vector_b)
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            op_a = qt.ket2dm(spin_a[i])
            op_b = qt.ket2dm(spin_b[j])
            p = (qt.tensor(op_a, op_b) * state).tr()
            if np.imag(p) > 0.1:
                raise(ValueError)
            prob.append(np.real(p))
    return prob


def random_joint_vectors(n):
    """Generate a list of random 3D unit vectors."""
    a, b = [], []
    for i in range(n):
        vector_a = random_unit_vector(3)
        vector_b = random_unit_vector(3)
        a.append(vector_a)
        b.append(vector_b)
    return (a, b)


def generate_dataset(state, n):
    """Generate a dataset for the training of the NN by generating n random unit vectors.

    Args:
        state (qt.dm): The two-qubits entangled state
        n (int): The number of measurements to be generated

    Returns:
        dataframe: A dataframe with the a and b input vectors, output A and B, and the respective probabilities
    """
    a, b, p = [], [], []
    ct = 0
    for i in range(n - 2):
        vector_a = random_unit_vector(3)
        vector_b = random_unit_vector(3)
        prob = probability_dict(state, vector_a, vector_b)
        a.append(vector_a)
        b.append(vector_b)
        p.append(prob)
        ct += 1
        if ct == 100:
            # print(f'Progress is {i / n * 100 :.2f}%.')
            ct = 0

    # Adds two special cases to the distribution
    vector_a = [0, 0, 1]
    vector_b = [0, 0, -1]
    prob = probability_dict(state, vector_a, vector_b)
    a.append(vector_a)
    b.append(vector_b)
    p.append(prob)

    vector_a = [0, 0, 1]
    vector_b = [0, 0, 1]
    prob = probability_dict(state, vector_a, vector_b)
    a.append(vector_a)
    b.append(vector_b)
    p.append(prob)

    # print('Finished.')

    ax = [val[0] for val in a for _ in range(9)]
    ay = [val[1] for val in a for _ in range(9)]
    az = [val[2] for val in a for _ in range(9)]
    bx = [val[0] for val in b for _ in range(9)]
    by = [val[1] for val in b for _ in range(9)]
    bz = [val[2] for val in b for _ in range(9)]
    l = []
    for d in p:
        for (k, v) in d.items():
            l.append([k[0], k[1], v])
    l = np.transpose(l)
    A = l[0]
    B = l[1]
    P = l[2]
    index = [val for val in range(n) for _ in range(9)]
    return pd.DataFrame({'ax': ax, 'ay': ay, 'az': az, 'bx': bx, 'by': by, 'bz': bz, 'A': A, 'B': B,
                         'probability': P, 'index': index})


def generate_dataset_from_vectors(state, a, b):
    """Generate a dataset for the training of the NN from a list of vectors.

    Args:
        state (qt.dm): The two-qubits entangled state.
        a (list): A list of 3D unit vectors.
        b (list): A list of 3D unit vectors. Must be of the same size as a.

    Returns:
        dataframe: A dataframe with the a and b input vectors, output A and B, and the respective probabilities
    """
    p = []
    for i in range(len(a)):
        vec_a = a[i]
        vec_b = b[i]
        prob = probability_dict(state, vec_a, vec_b)
        p.append(prob)
    ax = [val[0] for val in a for _ in range(4)]
    ay = [val[1] for val in a for _ in range(4)]
    az = [val[2] for val in a for _ in range(4)]
    bx = [val[0] for val in b for _ in range(4)]
    by = [val[1] for val in b for _ in range(4)]
    bz = [val[2] for val in b for _ in range(4)]
    l = []
    for d in p:
        for (k, v) in d.items():
            l.append([k[0], k[1], v])
    l = np.transpose(l)
    A = l[0]
    B = l[1]
    P = l[2]
    index = [val for val in range(len(a)) for _ in range(4)]
    return pd.DataFrame({'ax': ax, 'ay': ay, 'az': az, 'bx': bx, 'by': by, 'bz': bz, 'A': A, 'B': B,
                         'probability': P, 'index': index})


def nme_state(theta):
    """Generate a non-maximally entangled ket state defined by cos(theta)|-1-1> + sin(theta)|11>."""
    a = np.cos(theta)
    b = np.sin(theta)
    return a * qt.tensor(qt.basis(3, 0), qt.basis(3, 0)) + b * qt.tensor(qt.basis(3, 2), qt.basis(3, 2))


def mixed_separable(p=0.5):
    """Generate a mixed separable density matrix defined by p|-1-1><-1-1| + (1-p)|11><11|."""
    return p * qt.ket2dm(qt.tensor(qt.basis(3, 0), qt.basis(3, 0))) + (1-p) * qt.ket2dm(qt.tensor(qt.basis(3, 2), qt.basis(3, 2)))


def CHSH_measurements():
    """Generate vectors for CHSH measurements settings."""
    vec_1 = [0, 0, 1]
    vec_2 = [1, 0, 0]
    vec_3 = [1/np.sqrt(2), 0, 1/np.sqrt(2)]
    vec_4 = [-1/np.sqrt(2), 0, 1/np.sqrt(2)]

    a = [vec_1, vec_1, vec_2, vec_2]
    b = [vec_3, vec_4, vec_3, vec_4]
    return (a, b)


def CHSH_measurements_extended():
    """Generate vectors for CHSH measurements. Extended to include a mirrored and inverted version."""
    vec_a1 = [0, 0, 1]
    vec_a2 = [1, 0, 0]
    vec_b1 = [1/np.sqrt(2), 0, 1/np.sqrt(2)]
    vec_b2 = [-1/np.sqrt(2), 0, 1/np.sqrt(2)]

    vec_a3 = [0, 0, -1]
    vec_a4 = [0, 1, 0]
    vec_b3 = [0, 1/np.sqrt(2), -1/np.sqrt(2)]
    vec_b4 = [0, -1/np.sqrt(2), -1/np.sqrt(2)]

    a = [vec_a1, vec_a1, vec_a2, vec_a2, vec_a3, vec_a3, vec_a4, vec_a4]
    b = [vec_b1, vec_b2, vec_b1, vec_b2, vec_b3, vec_b4, vec_b3, vec_b4]
    return (a, b)


def correlated_measurements(n=2):
    """Create a joint measurements list by combining two lists of n vectors from each party."""
    vec_a1 = [0, 0, 1]
    vec_a2 = [1, 0, 0]
    vec_b1 = [1/np.sqrt(2), 0, 1/np.sqrt(2)]
    vec_b2 = [-1/np.sqrt(2), 0, 1/np.sqrt(2)]

    vec_a_add, vec_b_add = random_joint_vectors(n - 2)

    vec_a_list = [vec_a1, vec_a2] + vec_a_add
    vec_b_list = [vec_b1, vec_b2] + vec_b_add

    return combine_measurements(vec_a_list, vec_b_list)


def combine_measurements(vec_a_list, vec_b_list):
    """Combine two lists of measurements, one from each party, into a joint measurement list."""
    alice_list = []
    bob_list = []
    for vec_a in vec_a_list:
        for vec_b in vec_b_list:
            alice_list.append(vec_a)
            bob_list.append(vec_b)
    return alice_list, bob_list


def read_from_vector_dataset(filename):
    """Reads the vectors from a .csv dataset file containing 3D vectors.

    Args:
        filename (str): The filename containing the vectors

    Returns:
        tuple: A tuple containing two items, the list of vectors for Alice and Bob.
    """
    array = np.genfromtxt(filename, delimiter=",")
    vec_alice = []
    vec_bob = []
    for row in array:
        vec_alice.append(row[:3])
        vec_bob.append(row[3:])
    return (vec_alice, vec_bob)
