import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model_testing import *
from neural_network_util import build_model_comm, comm_customLoss_multiple, build_NewModel, build_NewModel_Afix
import config
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import qutip as qt
import pickle
from distribution_generator import *
from training import *

config.shuffle_epochs = 1
config.epochs = 100

folder_name = "NewModel\\TV\\5pi-32_singlet_1\\"
state = qt.ket2dm(nme_singlet(5*np.pi/32))

minimas = []
histories = []
K.clear_session()

# LOAD OLD MODEL
model = keras.models.load_model("NewModel\\TV\\3pi-16_singlet_2\\3pi_16_model.h5", compile=False)

config.epochs = 1
vec_alice, vec_bob = maximum_violation_measurements_extended(5*np.pi/32, n=config.training_size)
dataset_df = generate_dataset_from_vectors(state, vec_alice, vec_bob)
dataset_df.to_csv(folder_name + "dummy.csv")
train(model, folder_name + "dummy.csv", save=False, lr=1e-7, loss=comm_customLoss_multiple)
print("Model finished.")

config.epochs = 100
minima, history = train_generator(model, create_generator(state), save=True,
                        save_name=folder_name + '5pi_32_model.h5', loss=comm_customLoss_multiple, steps=50)
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
