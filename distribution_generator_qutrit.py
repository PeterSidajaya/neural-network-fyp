import qutip as qt
import numpy as np
import pandas as pd
import math

"""This file contains the functions needed to create the datasets."""

def random_pure_state(dim):
    U_real = np.matrix(np.random.normal(size=(dim,dim)))
    U_imag = np.matrix(np.random.normal(size=(dim,dim)))

    U, R = np.linalg.qr(U_real + 1j* U_imag, mode='complete')
    
    d = np.matrix(np.zeros((dim, 1)))
    d[0,0] = 1
    return dephase(U * d)

    d = np.matrix(np.zeros((dim, dim)))
    d[0,0] = 1
    
    X = U * d * U.H
    eigval, eigvec = np.linalg.eig(X)

    for i in range(dim):
        if np.abs(eigval[i] - 1) < 10**-10:
            vec = eigvec[:,i]
            break
    
    return dephase(vec)

def random_orthonormal_basis(dim):
    X = np.matrix(np.ndarray((dim,0)))
    for _ in range(dim):
        X = np.matrix(np.hstack((X,random_pure_state(dim))))
    Q = np.linalg.qr(X)[0]
    vector_set = []
    for i in range(dim):
        v = Q[:,i]
        vector_set.append(qt.Qobj(dephase(v)).unit())
    return vector_set

def dephase(vec):
    A = vec[0,0]
    phi = np.arctan2(np.imag(A), np.real(A))
    return np.exp(-1j*phi)*vec


def parametrise(vector_list):
    params = []
    for vector in vector_list:
        for i in range(3):
            overlap = vector.overlap(qt.basis(3,i))
            params.append(np.real(overlap))
            if i > 0:
                params.append(np.imag(overlap))
    return params



def probability_dict(state, vector_set_a, vector_set_b):
    """Returns the probabilities of a joint measurement defined by two unit vectors on an entangled state."""
    prob = {}
    for i in range(3):
        for j in range(3):
            op_a = qt.ket2dm(vector_set_a[i])
            op_b = qt.ket2dm(vector_set_b[j])
            prob[i, j] = (qt.tensor(op_a, op_b)
                                          * state).tr()
    return prob


def probability_list(state, vector_set_a, vector_set_b):
    """Returns the probabilities of a joint measurement defined by two unit vectors on an entangled state."""
    prob = []
    for i in range(3):
        for j in range(3):
            op_a = qt.ket2dm(vector_set_a[i])
            op_b = qt.ket2dm(vector_set_b[j])
            p = (qt.tensor(op_a, op_b) * state).tr()
            if np.imag(p) > 0.1:
                raise(ValueError)
            prob.append(np.real(p))
    return prob

