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


folder_name = "new-LHV\\pi-4_100\\"
state = qt.ket2dm(nme_state(np.pi/4))


model = keras.models.load_model(folder_name + "pi_4_model.h5", compile=False)
print(validate(model, state))
