# Import the required packages
import numpy as np

def dot(vec_1, vec_2):
    """Dot product for two vectors."""
    return vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1] + vec_1[2] * vec_2[2]


def outer_product(vec_1, vec_2):
    """Do an outer product then flatten the vector. Used for mixing the probabilities."""
    return np.outer(vec_1, vec_2).flatten()

def alice_protocol_1(vec_alice, lhv):
    """First protocol of Alice. Output is in the form of (P(A=+1), P(A=-1))"""
    prob = 1/2 - 1/2 * np.sign(dot(vec_alice, lhv[0] + lhv[1]))
    return np.array([prob, 1-prob])


def alice_protocol_2(vec_alice, lhv):
    """Second protocol of Alice. Output is in the form of (P(A=+1), P(A=-1))"""
    prob = 1/2 - 1/2 * np.sign(dot(vec_alice, lhv[0] - lhv[1]))
    return np.array([prob, 1-prob])


def bob_protocol_1(vec_bob, lhv):
    """First protocol of Bob. Output is in the form of (P(A=+1), P(A=-1))"""
    prob = 1/2 + 1/2 * np.sign(dot(vec_bob, lhv[0] + lhv[1]))
    return np.array([prob, 1-prob])


def bob_protocol_2(vec_bob, lhv):
    """Second protocol of Bob. Output is in the form of (P(A=+1), P(A=-1))"""
    prob = 1/2 + 1/2 * np.sign(dot(vec_bob, lhv[0] - lhv[1]))
    return np.array([prob, 1-prob])

def comm_protocol(vec_alice, lhv):
    """Protocol for the communication bit. The output is the probability of choosing the first protocols."""
    M1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    M2 = M1
    return 1/2 + 1/4 * np.sign(mdot(vec_alice, lhv[0], M1)) * np.sign(mdot(vec_alice, lhv[1], M1)) \
        + 1/4 * np.sign(mdot(vec_alice, lhv[0], M2)) * np.sign(mdot(vec_alice, lhv[1], M2))

def mdot(vec_1, vec_2, M):
    """Dot product for two vectors."""
    vec_1 = np.reshape(vec_1, (-1,3))
    vec_2 = np.reshape(vec_2, (3,-1))
    return np.matmul(vec_1, np.matmul(M, vec_2))[0][0]