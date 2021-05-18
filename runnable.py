import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from grapher import plot_dataset, plot_measurements
from training import *
from distribution_generator import *
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


config.shuffle_epochs = 100
config.training_size = 48
config.epochs = 2000


"""
filename = "model_save_12052021\\dataset.csv"

losses_with_comm_nme = []
histories_with_comm_nme = []
for i in range(1):
    K.clear_session()
    model = build_model()
    model.compile(loss=customLoss_multiple, optimizer=config.optimizer, metrics=[])
    minima, history = train_model_history(filename)
    losses_with_comm_nme.append(minima)
    histories_with_comm_nme.append(history)
    if minima < config.cutoff:
        break

filename = 'model_save_12052021\\nme_loss_without_comm.csv'
np.savetxt(filename, losses_with_comm_nme)

count = 0
for run in histories_with_comm_nme:
    filename = 'model_save_12052021\\history_without_comm_nme_' + \
        str(count+1) + '.csv'
    np.savetxt(filename, histories_with_comm_nme[count])
    count += 1
"""


model = keras.models.load_model("model_save_12052021\\pi_8_model_retrain.h5", compile=False)
# distr = comm_distr(model, n = 4096)
# distr.to_csv("model_save_12052021\\distr.csv")
# plot_comm_distr(distr, type='spherical')
print(evaluate(model, "model_save_12052021\\dataset.csv"))
print(validate(model, qt.ket2dm(nme_state(np.pi/8))))
