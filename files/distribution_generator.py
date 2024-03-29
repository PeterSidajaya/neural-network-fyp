import qutip as qt
import numpy as np
import pandas as pd
import math

"""This file contains the functions needed to create the datasets."""


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


def random_semicircle_vector():
    """Generate a 2D random vector from a semicircle."""
    components = [np.random.normal() for i in range(2)]
    r = math.sqrt(sum(x*x for x in components))
    v = [x/r for x in components]
    return [abs(v[0]), v[1]]


def operator_dot(vector):
    """Calculate the dot product of a vector with the Pauli matrices vector."""
    return vector[0] * qt.sigmax() + vector[1] * qt.sigmay() + vector[2] * qt.sigmaz()


def probability(state, vector_a, vector_b):
    """Returns the probabilities of a joint measurement defined by two unit vectors on an entangled state."""
    prob = {}
    for i in range(2):
        for j in range(2):
            op_a = 0.5 * (qt.identity(2) + (-1) ** i * operator_dot(vector_a))
            op_b = 0.5 * (qt.identity(2) + (-1) ** j * operator_dot(vector_b))
            prob[(-1) ** i, (-1) ** j] = (qt.tensor(op_a, op_b)
                                          * state).tr()
    return prob


def probability_list(state, vector_a, vector_b):
    """Returns the probabilities of a joint measurement defined by two unit vectors on an entangled state."""
    prob = []
    for i in range(2):
        for j in range(2):
            op_a = 0.5 * (qt.identity(2) + (-1) ** i * operator_dot(vector_a))
            op_b = 0.5 * (qt.identity(2) + (-1) ** j * operator_dot(vector_b))
            prob.append((qt.tensor(op_a, op_b) * state).tr())
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
    """DEPRECATED. Generate a dataset for the training of the NN by generating n random unit vectors.

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
        prob = probability(state, vector_a, vector_b)
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
    prob = probability(state, vector_a, vector_b)
    a.append(vector_a)
    b.append(vector_b)
    p.append(prob)

    vector_a = [0, 0, 1]
    vector_b = [0, 0, 1]
    prob = probability(state, vector_a, vector_b)
    a.append(vector_a)
    b.append(vector_b)
    p.append(prob)

    # print('Finished.')

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
    index = [val for val in range(n) for _ in range(4)]
    return pd.DataFrame({'ax': ax, 'ay': ay, 'az': az, 'bx': bx, 'by': by, 'bz': bz, 'A': A, 'B': B,
                         'probability': P, 'index': index})


def generate_dataset_from_vectors(state, a, b):
    """DEPRECATED. Generate a dataset for the training of the NN from a list of vectors.

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
        prob = probability(state, vec_a, vec_b)
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
    """Generate a non-maximally entangled ket state defined by cos(theta)|00> + sin(theta)|11>."""
    a = np.cos(theta)
    b = np.sin(theta)
    return a * qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + b * qt.tensor(qt.basis(2, 1), qt.basis(2, 1))


def nme_singlet(theta):
    """Generate a non-maximally entangled ket state defined by cos(theta)|01> - sin(theta)|10>."""
    a = np.cos(theta)
    b = np.sin(theta)
    return a * qt.tensor(qt.basis(2, 0), qt.basis(2, 1)) - b * qt.tensor(qt.basis(2, 1), qt.basis(2, 0))


def mixed_separable(p=0.5):
    """Generate a mixed separable density matrix defined by p|00><00| + (1-p)|11><11|."""
    return p * qt.ket2dm(qt.tensor(qt.basis(2, 0), qt.basis(2, 0))) + (1-p) * qt.ket2dm(qt.tensor(qt.basis(2, 1), qt.basis(2, 1)))


def generate_werner(n=8, start=0, end=1, step=10, a=None, b=None):
    """DEPRECATED. Generate a series of werner state datasets
    Args:
        n: number of measurements (default 8)
        start: starting werner state parameter (default 0)
        end: ending werner state parameter (default 1)
        step: number of datasets to be generated (default 10)
        a: list of vectors for Alice (default None)
        b: list of vectors for Bob (default None)
    """
    iden = 1/4 * qt.tensor(qt.identity(2), qt.identity(2))
    bell = qt.ket2dm(qt.bell_state('11'))
    count = 0
    # Check if vectors are passed. If not, generate randomly
    if a == None or b == None:
        a, b = random_joint_vectors(n)
    for w in np.linspace(start, end, step):
        state = w * bell + (1-w) * iden
        filename = 'datasets\\dataset_werner_state_' + str(count) + '.csv'
        generate_dataset_from_vectors(state, a, b).to_csv(filename)
        count += 1


def generate_mixed(n=8, start=0, end=1, step=10, a=None, b=None):
    """DEPRECATED. Generate a series of mixed state datasets.
    Keyword arguments:
        n: number of measurements (default 8)
        start: starting mixed state parameter (default 0)
        end: ending mixed state parameter (default 1)
        step: number of datasets to be generated (default 10)
        a: list of vectors for Alice (default None)
        b: list of vectors for Bob (default None)
    """
    count = 0
    # Check if vectors are passed. If not, generate randomly
    if a == None or b == None:
        a, b = random_joint_vectors(n)
    for p in np.linspace(start, end, step):
        state = mixed_separable(p=p)
        filename = 'datasets\\dataset_mixed_separable_state_' + \
            str(count) + '.csv'
        generate_dataset_from_vectors(state, a, b).to_csv(filename)
        count += 1


def generate_random_settings(n=8, state_type='max_entangled'):
    """DEPRECATED. Generate a dataset for n random measurements of a state"""
    if state_type == 'max_entangled':
        state = qt.ket2dm(nme_state(np.pi/4))
        filename = 'datasets\\dataset_maximally_entangled_state.csv'
    elif state_type == 'entangled':
        state = qt.ket2dm(nme_state(np.pi/8))
        filename = 'datasets\\dataset_non_maximally_entangled_pi8_state.csv'
    elif state_type == 'product':
        state = qt.ket2dm(nme_state(0))
        filename = 'datasets\\dataset_product_state.csv'
    elif state_type == 'mixed':
        state = mixed_separable()
        filename = 'datasets\\dataset_mixed_separable_state.csv'
    generate_dataset(state, n).to_csv(filename)
    return filename


def CHSH_measurements():
    """DEPRECATED. Generate vectors for CHSH measurements settings."""
    vec_1 = [0, 0, 1]
    vec_2 = [1, 0, 0]
    vec_3 = [1/np.sqrt(2), 0, 1/np.sqrt(2)]
    vec_4 = [-1/np.sqrt(2), 0, 1/np.sqrt(2)]

    a = [vec_1, vec_1, vec_2, vec_2]
    b = [vec_3, vec_4, vec_3, vec_4]
    return (a, b)


def CHSH_measurements_extended():
    """DEPRECATED. Generate vectors for CHSH measurements. Extended to include a mirrored and inverted version."""
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


def maximum_violation_measurements(theta):
    """DEPRECATED. Generate 8 pairs of maximally nonlocal (for NME states) measurements."""
    kai = np.arccos(1/np.sqrt(1+np.sin(2*theta)**2))
    vec_a1 = [0, 0, 1]
    vec_a2 = [1, 0, 0]
    vec_b1 = np.multiply(np.cos(kai), vec_a1) + \
        np.multiply(np.sin(kai), vec_a2)
    vec_b2 = np.multiply(np.cos(kai), vec_a1) - \
        np.multiply(np.sin(kai), vec_a2)

    vec_a3 = [0, 0, -1]
    vec_a4 = [0, 1, 0]
    vec_b3 = np.multiply(np.cos(kai), vec_a3) + \
        np.multiply(np.sin(kai), vec_a4)
    vec_b4 = np.multiply(np.cos(kai), vec_a3) - \
        np.multiply(np.sin(kai), vec_a4)

    a = [vec_a1, vec_a1, vec_a2, vec_a2, vec_a3, vec_a3, vec_a4, vec_a4]
    b = [vec_b1, vec_b2, vec_b1, vec_b2, vec_b3, vec_b4, vec_b3, vec_b4]
    return (a, b)


def maximum_violation_measurements_extended(theta, n=8):
    """DEPRECATED. Generate n number of measurements, 8 of which is generated from the
    maximum_violation_measurements() function.

    Args:
        theta (float): The theta of the non-maximally entangled state.
        n (int, optional): The number of measurements to be done. Defaults to 8.

    Returns:
        tuple: A tuple containing two items, the list of vectors for Alice and Bob.
    """
    (vec_alice, vec_bob) = maximum_violation_measurements(theta)
    n_add = n-8
    (vec_alice_add, vec_bob_add) = random_joint_vectors(n_add)
    vec_alice += vec_alice_add
    vec_bob += vec_bob_add
    return (vec_alice, vec_bob)


def correlated_measurements(theta, n=2):
    """DEPRECATED. Create a joint measurements list by combining two lists of n vectors from each party."""
    kai = np.arccos(1/np.sqrt(1+np.sin(2*theta)**2))
    vec_a1 = [0, 0, 1]
    vec_a2 = [1, 0, 0]
    vec_b1 = np.multiply(np.cos(kai), vec_a1) + \
        np.multiply(np.sin(kai), vec_a2)
    vec_b2 = np.multiply(np.cos(kai), vec_a1) - \
        np.multiply(np.sin(kai), vec_a2)

    vec_a_add, vec_b_add = random_joint_vectors(n - 2)

    vec_a_list = [vec_a1, vec_a2] + vec_a_add
    vec_b_list = [vec_b1, vec_b2] + vec_b_add

    return combine_measurements(vec_a_list, vec_b_list)


def combine_measurements(vec_a_list, vec_b_list):
    """DEPRECATED. Combine two lists of measurements, one from each party, into a joint measurement list."""
    alice_list = []
    bob_list = []
    for vec_a in vec_a_list:
        for vec_b in vec_b_list:
            alice_list.append(vec_a)
            bob_list.append(vec_b)
    return alice_list, bob_list


def read_from_vector_dataset(filename):
    """DEPRECATED. Reads the vectors from a .csv dataset file containing 3D vectors.

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
