import numpy as np
import qutip as qt
import os
import tensorflow.keras as keras
import config
from model_testing import predict

"""This file is used to evaluate the errors in the model. The plotting is done in a separate file
that is not included in this repo. This is the newest file, and the most messy. Sorry :)
"""

os.system('clear')
np.set_printoptions(linewidth=np.inf)

def nme_state(theta):
    """Generate a non-maximally entangled ket state defined by cos(theta)|00> + sin(theta)|11>."""
    a = np.cos(theta)
    b = np.sin(theta)
    return a * qt.tensor(qt.basis(2, 0), qt.basis(2, 1)) - b * qt.tensor(qt.basis(2, 1), qt.basis(2, 0))

folder_name = "qubits-new/qubit-7pi-32/"
model = keras.models.load_model(folder_name + "7pi-32_model.h5", compile=False)
state = qt.ket2dm(nme_state(7*np.pi/32))
filename = "qubit-7pi-32-benchmark-ns.csv"

def random_unit_vectors(m, n):
    """Generate m random vectors of n dimensions."""
    array = np.random.normal(size=(m, n))
    norm = np.linalg.norm(array, axis=1)
    return array / norm[:, None]

def operator_dot(vector):
    """Calculate the dot product of a vector with the Pauli matrices vector."""
    return vector[0] * qt.sigmax() + vector[1] * qt.sigmay() + vector[2] * qt.sigmaz()

def probability_list(state, vector_a, vector_b):
    """Returns the probabilities of a joint measurement defined by two unit vectors on an entangled state."""
    prob = []
    for i in range(2):
        for j in range(2):
            op_a = 0.5 * (qt.identity(2) + (-1) ** i * operator_dot(vector_a))
            op_b = 0.5 * (qt.identity(2) + (-1) ** j * operator_dot(vector_b))
            prob.append((qt.tensor(op_a, op_b) * state).tr())
    return np.round(prob, 5)

def kl(p,q):
    p = np.clip(p,np.finfo(np.float32).eps,1)
    q = np.clip(q,np.finfo(np.float32).eps,1)
    return np.sum(p*np.log(p/q))

def tvd(p,q):
    return 1/2*np.sum(np.abs(p-q))


x = np.array([1, 0, 0])
y = np.array([0, 1, 0])
z = np.array([0, 0, 1])

def num_rounds(kl_error):
    return -np.log(0.05)/kl_error

UP = "\x1B[3A"
CLR = "\x1B[0K"

res = np.zeros((1000,5))

for __ in range(1000):

    A = random_unit_vectors(1,3)[0]
    B = random_unit_vectors(1,3)[0]

    config.training_size = 1
    config.LHV_size = 10000
    result = predict(model,np.array([np.array([A,B]).flatten()]))

    res[__,:] = np.array([kl(result, probability_list(state, A, B)), kl(probability_list(state, A, B), result), tvd(probability_list(state, A, B), result),
                     num_rounds(kl(result, probability_list(state, A, B))), num_rounds(kl(probability_list(state, A, B), result))])
    print(f'Avg: {np.average(res[:__+1,:],axis=0)}\nStd: {np.std(res[:__+1,:],axis=0)}\n')

np.savetxt(filename, res, delimiter=',')

