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
config.epochs = 150

folder_name = "NewModel\\pi-16_singlet_1\\"
state = qt.ket2dm(nme_singlet(np.pi/16))

"""
# FIRST RUN

config.LHV_size = 600		        # number of LHV for evaluation
config.training_size = 300		    # number of measurement settings for one training step

filename_1 = folder_name + "dataset_1.csv"

vec_alice, vec_bob = maximum_violation_measurements_extended(np.pi/16, n=6000)
dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
dataset_df.to_csv(filename_1)

minimas_1 = []
histories_1 = []
K.clear_session()

# LOAD OLD MODEL
model = build_NewModel()
print("Model finished.")

minima, history = train(model, filename_1, save=True,
                        save_name=folder_name + 'pi_16_model_1.h5', loss=comm_customLoss_multiple)
minimas_1.append(minima)
histories_1.append(history)

model.save_weights(folder_name + 'weights_1.h5')
symbolic_weights = getattr(model.optimizer, 'weights')
weight_values = K.batch_get_value(symbolic_weights)
with open(folder_name + 'optimizer_1.pkl', 'wb') as f:
    pickle.dump(weight_values, f)

save_name = folder_name + 'loss_1.csv'
np.savetxt(save_name, minimas_1)
save_name = folder_name + 'history_1.csv'
np.savetxt(save_name, histories_1[0])

print("Training finished")


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# SECOND RUN

config.LHV_size = 750		        # number of LHV for evaluation
config.training_size = 300		    # number of measurement settings for one training step

filename_2 = folder_name + "dataset_2.csv"

vec_alice, vec_bob = maximum_violation_measurements_extended(np.pi/16, n=9000)
dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
dataset_df.to_csv(filename_2)

minimas_2 = []
histories_2 = []
K.clear_session()

# LOAD OLD MODEL
model = keras.models.load_model("NewModel\\pi-16_singlet_1\\pi_16_model_1.h5", compile=False)

config.epochs = 1
vec_alice, vec_bob = maximum_violation_measurements_extended(np.pi/16, n=config.training_size)
dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
dataset_df.to_csv(folder_name + "dummy.csv")
train(model, folder_name + "dummy.csv", save=False, lr=1e-10, loss=comm_customLoss_multiple)
print("Model finished.")

# LOAD OLD WEIGHTS
with open('NewModel\\pi-16_singlet_1\\optimizer_1.pkl', 'rb') as f:
    weight_values = pickle.load(f)
model.optimizer.set_weights(weight_values)

# SET EPOCHS
config.epochs = 70
minima, history = train(model, filename_2, save=True,
                        save_name=folder_name + 'pi_16_model_2.h5', loss=comm_customLoss_multiple)
minimas_2.append(minima)
histories_2.append(history)

model.save_weights(folder_name + 'weights_2.h5')
symbolic_weights = getattr(model.optimizer, 'weights')
weight_values = K.batch_get_value(symbolic_weights)
with open(folder_name + 'optimizer_2.pkl', 'wb') as f:
    pickle.dump(weight_values, f)

save_name = folder_name + 'loss_2.csv'
np.savetxt(save_name, minimas_2)
save_name = folder_name + 'history_2.csv'
np.savetxt(save_name, histories_2[0])

print("Training finished")
"""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# THIRD RUN

config.LHV_size = 1000		        # number of LHV for evaluation
config.training_size = 300		    # number of measurement settings for one training step

filename_3 = folder_name + "dataset_3.csv"

vec_alice, vec_bob = maximum_violation_measurements_extended(np.pi/16, n=12000)
dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
dataset_df.to_csv(filename_3)

minimas_3 = []
histories_3 = []
K.clear_session()

# LOAD OLD MODEL
model = keras.models.load_model("NewModel\\pi-16_singlet_1\\pi_16_model_2.h5", compile=False)

config.epochs = 1
vec_alice, vec_bob = maximum_violation_measurements_extended(np.pi/16, n=config.training_size)
dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
dataset_df.to_csv(folder_name + "dummy.csv")
train(model, folder_name + "dummy.csv", save=False, lr=1e-10, loss=comm_customLoss_multiple)
print("Model finished.")

# LOAD OLD WEIGHTS
with open('NewModel\\pi-16_singlet_1\\optimizer_2.pkl', 'rb') as f:
    weight_values = pickle.load(f)
model.optimizer.set_weights(weight_values)

# SET EPOCHS
config.epochs = 30
minima, history = train(model, filename_3, save=True,
                        save_name=folder_name + 'pi_16_model_3.h5', loss=comm_customLoss_multiple)
minimas_3.append(minima)
histories_3.append(history)

model.save_weights(folder_name + 'weights_3.h5')
symbolic_weights = getattr(model.optimizer, 'weights')
weight_values = K.batch_get_value(symbolic_weights)
with open(folder_name + 'optimizer_3.pkl', 'wb') as f:
    pickle.dump(weight_values, f)

save_name = folder_name + 'loss_3.csv'
np.savetxt(save_name, minimas_3)
save_name = folder_name + 'history_3.csv'
np.savetxt(save_name, histories_3[0])

print("Training finished")