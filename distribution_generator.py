import qutip as qt
import numpy as np
import pandas as pd
import math
import os
import sys
import getopt


def random_vector(n):
    components = [np.random.normal() for i in range(n)]
    r = math.sqrt(sum(x*x for x in components))
    v = [x/r for x in components]
    return v


def operator_dot(vector):
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


def generate_random_vectors(n):
    a, b = [], []
    for i in range(n):
        vector_a = random_vector(3)
        vector_b = random_vector(3)
        a.append(vector_a)
        b.append(vector_b)
    return (a, b)


def generate_dataset(state, n):
    """Generate a dataset for the training of the NN

    Args:
        state (qt.dm): The two-qubits entangled state
        n (int): The number of measurements to be generated

    Returns:
        dataframe: A dataframe with the a and b input vectors, output A and B, and the respective probabilities
    """
    a, b, p = [], [], []
    ct = 0
    for i in range(n - 2):
        vector_a = random_vector(3)
        vector_b = random_vector(3)
        prob = probability(state, vector_a, vector_b)
        a.append(vector_a)
        b.append(vector_b)
        p.append(prob)
        ct += 1
        if ct == 100:
            print(f'Progress is {i / n * 100 :.2f}%.')
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

    print('Finished.')

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
    """Generate a non-maximally entangled state defined by cos(theta)|00> + sin(theta)|11>."""
    a = np.cos(theta)
    b = np.sin(theta)
    return a * qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + b * qt.tensor(qt.basis(2, 1), qt.basis(2, 1))


def mixed_separable(p=0.5):
    return p * qt.ket2dm(qt.tensor(qt.basis(2, 0), qt.basis(2, 0))) + (1-p) * qt.ket2dm(qt.tensor(qt.basis(2, 1), qt.basis(2, 1)))


def usage():
    print("usage: " +
          sys.argv[0] + " -i type_of_state (product, max_entangled, non_max_entangled, mixed_separable)")


def generate_werner(n=8, step=10, a=None, b=None):
    """Generate a series of werner state datasets, different measurements"""
    iden = 1/4 * qt.tensor(qt.identity(2), qt.identity(2))
    bell = qt.ket2dm(qt.bell_state('11'))
    count = 0
    if a == None and b == None:
        a, b = generate_random_vectors(n)
    for w in np.linspace(0, 1, step):
        state = w * bell + (1-w) * iden
        filename = 'datasets\dataset_werner_state_' + str(count) + '.csv'
        generate_dataset_from_vectors(state, a, b).to_csv(filename)
        count += 1


def generate_mixed(n=8, step=10, a=None, b=None):
    """Generate a series of mixed state datasets, different measurements"""
    count = 0
    if a == None and b == None:
        a, b = generate_random_vectors(n)
    for p in np.linspace(0, 1, step):
        state = mixed_separable(p=p)
        filename = 'datasets\dataset_mixed_separable_state_' + \
            str(count) + '.csv'
        generate_dataset_from_vectors(state, a, b).to_csv(filename)
        count += 1


def generate_different_settings(start=1, end=64, step=10, state_type='entangled'):
    """Generate a series of werner state datasets, different measurements"""
    if state_type == 'max_entangled':
        state = qt.ket2dm(nme_state(np.pi/4))
        base_filename = 'datasets\dataset_maximally_entangled_state_'
        generate_dataset(state, n).to_csv(filename)
    elif state_type == 'entangled':
        state = qt.ket2dm(nme_state(np.pi/8))
        base_filename = 'datasets\dataset_non_maximally_entangled_pi8_state_'
    count = 0
    for n in np.linspace(start, end, step):
        a, b = generate_random_vectors(4*int(n))
        filename = base_filename + str(count) + '.csv'
        generate_dataset_from_vectors(state, a, b).to_csv(filename)
        count += 1

def main():
    state_type = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-i':
            state_type = a
        else:
            assert False, "unhandled option"

    n = int(2 ** 10)

    if state_type == 'product':
        state = qt.ket2dm(nme_state(0))
        filename = 'dataset_product_state.csv'
        generate_dataset(state, n).to_csv(filename)
    elif state_type == 'max_entangled':
        state = qt.ket2dm(nme_state(np.pi/4))
        filename = 'dataset_maximally_entangled_state.csv'
        generate_dataset(state, n).to_csv(filename)
    elif state_type == 'entangled':
        state = qt.ket2dm(nme_state(np.pi/8))
        filename = 'dataset_non_maximally_entangled_pi8_state.csv'
        generate_dataset(state, n).to_csv(filename)
    elif state_type == 'mixed_separable':
        state = 0.7 * qt.ket2dm(nme_state(0)) + 0.3 * \
            qt.ket2dm(nme_state(np.pi/2))
        filename = 'dataset_mixed_separable_state.csv'
        generate_dataset(state, n).to_csv(filename)

    if state_type == None:
        usage()
        sys.exit(2)


if __name__ == "__main__":
    main()
