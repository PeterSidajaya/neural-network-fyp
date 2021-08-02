from grapher import *
from training import *
from distribution_generator import *
import pickle
import qutip as qt
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import config
from matplotlib import cm
from neural_network_util import build_model_comm, comm_customLoss_multiple
from model_testing import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


folder_name = "new-LHV\\pi-16_200_SV\\"
state = qt.ket2dm(nme_state(np.pi/16))


config.LHV_type = "vector"
model = keras.models.load_model(folder_name + "pi_16_model.h5", compile=False)
# print(validate(model, state, comm=True))

evaluate_marginals(model, np.pi/16, random_unit_vector(3),
                   random_unit_vector(3), singlet=False)
