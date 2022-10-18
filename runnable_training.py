import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from neural_network_util import *
import config
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import qutip as qt
import pickle
from distribution_generator import *
from training import *

folder_name = "qutrit\\"

minimas = []
histories = []
K.clear_session()

model = build_Model_v3()
keras.utils.plot_model(model, to_file='qutrit_v3.png',show_shapes=True, dpi=300)
print("Model finished.")

ket = (qt.tensor(qt.basis(3,0), qt.basis(3,0)) + qt.tensor(qt.basis(3,1), qt.basis(3,1)) + qt.tensor(qt.basis(3,2), qt.basis(3,2))).unit()
state = qt.ket2dm(ket)

minimas = []
histories = []
K.clear_session()

alice_set = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]
bob_set = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]
minima, history = train_generator(model, create_generator_limited(state, alice_set, bob_set, dim=3), save=True,
                        save_name=folder_name + 'spin_model.h5', loss=comm_customLoss_multiple, steps=50)
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
