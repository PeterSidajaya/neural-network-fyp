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
from distribution_generator_qutrit import *
from training import *

folder_name = "qutrit\\spin measurements - comm\\"
ket = (qt.tensor(qt.basis(3,0), qt.basis(3,0)) \
        + qt.tensor(qt.basis(3,1), qt.basis(3,1)) \
        + qt.tensor(qt.basis(3,2), qt.basis(3,2))).unit()
state = 0.75 * 1/9 * qt.tensor(qt.identity(3), qt.identity(3)) + 0.25 * qt.ket2dm(ket)

minimas = []
histories = []
K.clear_session()

model = build_NewModel()
print("Model finished.")

# # LOAD OLD MODEL
# model = keras.models.load_model(folder_name + '100 runs\\qutrit_no_noise.h5', compile=False)

# config.epochs = 1
# config.LHV_size = 1
# alice_measurements = [[1,0,0]]
# bob_measurements = [[1,0,0]]
# minima, history = train_generator(model,
#                                     create_generator_limited(state, alice_measurements, bob_measurements, dim=3),
#                                     loss=comm_customLoss_multiple, steps=1)

# # LOAD OLD WEIGHTS
# with open(folder_name + '100 runs\\optimizer.pkl', 'rb') as f:
#     weight_values = pickle.load(f)
# model.optimizer.set_weights(weight_values)

# print("Model finished.")

config.epochs = 100
config.LHV_size = 1000
alice_measurements = [[1,0,0], [0,0,1], [1/np.sqrt(2),0,1/np.sqrt(2)], [1/np.sqrt(2),0,-1/np.sqrt(2)],[0,1,0], [0,0,1], [0,1/np.sqrt(2),1/np.sqrt(2)], [0,1/np.sqrt(2),-1/np.sqrt(2)]]
bob_measurements = [[1,0,0], [0,0,1], [1/np.sqrt(2),0,1/np.sqrt(2)], [1/np.sqrt(2),0,-1/np.sqrt(2)],[0,1,0], [0,0,1], [0,1/np.sqrt(2),1/np.sqrt(2)], [0,1/np.sqrt(2),-1/np.sqrt(2)]]
minima, history = train_generator(model,
                                    create_generator_limited(state, alice_measurements, bob_measurements, dim=3), save=True,
                                    save_name=folder_name + 'qutrit_75_noise.h5', loss=comm_customLoss_multiple, steps=100)
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
