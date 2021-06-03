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

config.shuffle_epochs = 100
config.training_size = 64
config.epochs = 10000

folder_name = "Bell-inequality-search\\"
state = qt.ket2dm(nme_state(np.pi/16))


vec_alice, vec_bob = correlated_measurements(np.pi/16, n=8)

plot_measurements(vec_alice)
plot_measurements(vec_bob)

dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
filename = folder_name + "dataset.csv"
dataset_df.to_csv(filename)

minimas = []
histories = []
K.clear_session()

model = build_model_comm()

minima, history = train(model, filename, save=True,
                        save_name=folder_name + 'pi_16_model.h5')

minimas.append(minima)
histories.append(history)

model.save_weights(folder_name + 'weights.h5')
symbolic_weights = getattr(model.optimizer, 'weights')
weight_values = K.batch_get_value(symbolic_weights)
with open(folder_name + 'optimizer.pkl', 'wb') as f:
    pickle.dump(weight_values, f)

save_name = folder_name + 'nme_loss.csv'
np.savetxt(save_name, minimas)
save_name = folder_name + 'history_without_comm_nme.csv'
np.savetxt(save_name, histories[0])

print("Training finished")
