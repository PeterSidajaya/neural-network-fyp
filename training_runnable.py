import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model_testing import *
from neural_network_util import build_model_comm, comm_customLoss_multiple, build_model, customLoss_multiple
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

config.shuffle_epochs = 1
config.epochs = 50

folder_name = "symmetry\\pi-16_250_SV_singlet\\"
state = qt.ket2dm(nme_singlet(np.pi/16))


vec_alice, vec_bob = maximum_violation_measurements_extended(np.pi/16, n=12000)

dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
filename = folder_name + "dataset.csv"
dataset_df.to_csv(filename)

minimas = []
histories = []
K.clear_session()

model = keras.models.load_model("symmetry\\pi-16_200_SV_singlet\\pi_16_model.h5", compile=False)

config.epochs = 1
train(model, filename, save=False, lr=1e-6, loss=comm_customLoss_multiple)

config.epochs = 50
with open('symmetry\\pi-16_200_SV_singlet\\optimizer.pkl', 'rb') as f:
    weight_values = pickle.load(f)
model.optimizer.set_weights(weight_values)
minima, history = train(model, filename, save=True,
                        save_name=folder_name + 'pi_16_model.h5', loss=comm_customLoss_multiple)
minimas.append(minima)
histories.append(history)

model.save_weights(folder_name + 'weights.h5')
symbolic_weights = getattr(model.optimizer, 'weights')
weight_values = K.batch_get_value(symbolic_weights)
with open(folder_name + 'optimizer.pkl', 'wb') as f:
    pickle.dump(weight_values, f)

save_name = folder_name + 'loss.csv'
np.savetxt(save_name, minimas)
save_name = folder_name + 'history.csv'
np.savetxt(save_name, histories[0])

print("Training finished")
