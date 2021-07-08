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

folder_name = "new-LHV\\pi-16_150_SV_singlet\\"
state = qt.ket2dm(nme_singlet(np.pi/16))


config.LHV_type = "vector"
model = keras.models.load_model(folder_name + "pi_16_model.h5", compile=False)
# print(validate(model, state, comm=True))

distr = map_distr_SV_party(model, vec_alice=[1, 0, 0])
plot_comm_distr_vector(distr, type='scatter', color='comm', set_axes='alice', savename=None, show=True)
