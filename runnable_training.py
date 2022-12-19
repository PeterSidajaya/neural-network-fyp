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

folder_name = "qubit-pi-4/"

minimas = []
histories = []
K.clear_session()

model = build_Model_v3()
print("Model finished.")

state = qt.ket2dm(nme_singlet(np.pi/4))

minima, history = train_generator(model, create_generator(state), save=True,
                        save_name=folder_name + 'pi-4_model.h5', loss=comm_customLoss_multiple, steps=50)

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
