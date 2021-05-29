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


filename = "U5000R-100+300\\pi_8_model.h5"

state = qt.ket2dm(nme_state(np.pi/8))


model = keras.models.load_model(filename, compile=False)
distr = comm_distr(model, n = 4096)
plot_comm_distr(distr, type='spherical')