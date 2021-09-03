import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model_testing import *
from neural_network_util import build_model_comm, comm_customLoss_multiple, build_NewModel
import config
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import qutip as qt
import pickle
from distribution_generator import *
from training import *

config.shuffle_epochs = 1
config.epochs = 20

folder_name = "NewModel\\pi-16_fast_singlet_dot_model_320\\"
state = qt.ket2dm(nme_singlet(np.pi/16))

filename = folder_name + "dataset.csv"

vec_alice, vec_bob = maximum_violation_measurements_extended(np.pi/16, n=12000)
dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
dataset_df.to_csv(filename)

minimas = []
histories = []
K.clear_session()

# LOAD OLD MODEL
model = keras.models.load_model("NewModel\\pi-16_fast_singlet_dot_model\\pi_16_model.h5", compile=False)

config.epochs = 1
vec_alice, vec_bob = maximum_violation_measurements_extended(np.pi/16, n=300)
dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
dataset_df.to_csv(folder_name + "dummy.csv")
train(model, folder_name + "dummy.csv", save=False, lr=1e-7, loss=comm_customLoss_multiple)

print("Model finished.")
# LOAD OLD WEIGHTS
with open('NewModel\\pi-16_fast_singlet_dot_model\\optimizer.pkl', 'rb') as f:
    weight_values = pickle.load(f)
model.optimizer.set_weights(weight_values)

config.epochs = 20
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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

folder_name = "NewModel\\pi-16_fast_singlet_dot_model_340\\"
state = qt.ket2dm(nme_singlet(np.pi/16))

filename = folder_name + "dataset.csv"

vec_alice, vec_bob = maximum_violation_measurements_extended(np.pi/16, n=12000)
dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
dataset_df.to_csv(filename)

minimas = []
histories = []
K.clear_session()

# LOAD OLD MODEL
model = keras.models.load_model("NewModel\\pi-16_fast_singlet_dot_model_320\\pi_16_model.h5", compile=False)

config.epochs = 1
vec_alice, vec_bob = maximum_violation_measurements_extended(np.pi/16, n=300)
dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
dataset_df.to_csv(folder_name + "dummy.csv")
train(model, folder_name + "dummy.csv", save=False, lr=1e-7, loss=comm_customLoss_multiple)

print("Model finished.")
# LOAD OLD WEIGHTS
with open('NewModel\\pi-16_fast_singlet_dot_model_320\\optimizer.pkl', 'rb') as f:
    weight_values = pickle.load(f)
model.optimizer.set_weights(weight_values)

config.epochs = 20
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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

folder_name = "NewModel\\pi-16_fast_singlet_dot_model_360\\"
state = qt.ket2dm(nme_singlet(np.pi/16))

filename = folder_name + "dataset.csv"

vec_alice, vec_bob = maximum_violation_measurements_extended(np.pi/16, n=12000)
dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
dataset_df.to_csv(filename)

minimas = []
histories = []
K.clear_session()

# LOAD OLD MODEL
model = keras.models.load_model("NewModel\\pi-16_fast_singlet_dot_model_340\\pi_16_model.h5", compile=False)

config.epochs = 1
vec_alice, vec_bob = maximum_violation_measurements_extended(np.pi/16, n=300)
dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
dataset_df.to_csv(folder_name + "dummy.csv")
train(model, folder_name + "dummy.csv", save=False, lr=1e-7, loss=comm_customLoss_multiple)

print("Model finished.")
# LOAD OLD WEIGHTS
with open('NewModel\\pi-16_fast_singlet_dot_model_340\\optimizer.pkl', 'rb') as f:
    weight_values = pickle.load(f)
model.optimizer.set_weights(weight_values)

config.epochs = 20
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