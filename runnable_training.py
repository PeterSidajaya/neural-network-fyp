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

config.shuffle_epochs = 1
config.epochs = 10

folder_name = "qutrit\\"
ket = (qt.tensor(qt.basis(3,0), qt.basis(3,0)) \
        + qt.tensor(qt.basis(3,1), qt.basis(3,1)) \
        + qt.tensor(qt.basis(3,2), qt.basis(3,2))).unit()
state = 0.35 * 1/9 * qt.tensor(qt.identity(3), qt.identity(3)) + 0.65 * qt.ket2dm(ket)
model = build_NewModel_NC()
print("Model finished.")

minimas = []
histories = []
K.clear_session()

config.epochs = 25
minima, history = train_generator(model, create_generator(state, dim=3), save=True,
                        save_name=folder_name + 'qutrit_35_noise.h5', loss=customLoss_multiple, steps=100)
minimas.append(minima)
histories.append(history)

model.save_weights(folder_name + 'weights7.h5')
symbolic_weights = getattr(model.optimizer, 'weights')
weight_values = K.batch_get_value(symbolic_weights)
with open(folder_name + 'optimizer7.pkl', 'wb') as f:
    pickle.dump(weight_values, f)

save_name = folder_name + 'loss7.csv'
np.savetxt(save_name, minimas)
save_name = folder_name + 'history7.csv'
np.savetxt(save_name, histories[0])

print("Training finished")
