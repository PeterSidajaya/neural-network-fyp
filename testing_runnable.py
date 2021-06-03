import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model_testing import *
from neural_network_util import build_model_comm, comm_customLoss_multiple
from matplotlib import cm
import config
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.backend as K
import qutip as qt
import pickle
from distribution_generator import *
from training import *
from grapher import *


filename = "Bell-inequality-search\\search-1\\dataset.csv"
vec_alice, vec_bob = read_vectors(filename)
vec_alice = np.vstack([vec_alice[0:8], random_vector(3), random_vector(3)])
plot_measurements(vec_alice)
