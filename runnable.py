import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import config
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import qutip as qt
from distribution_generator import CHSH_measurements, CHSH_measurements_extended, generate_dataset_from_vectors, maximum_violation_measurements, maximum_violation_measurements_extended, nme_state
from training import communication_test, werner_run, train_model_comm_history, train_model_history, train_model_comm
from grapher import plot_dataset, plot_measurements


config.training_size = 24
config.epochs = 1500
config.shuffle_epochs = 100


# vector_a, vector_b = maximum_violation_measurements_extended(np.pi/8, n=24)

# np.savetxt('24_vector_list.csv', np.array(np.concatenate((vector_a, vector_b), axis=1)), delimiter=',')

filename1 = "overnight_training30032021\\overnight_training_me.csv"
# generate_dataset_from_vectors(qt.ket2dm(nme_state(np.pi/4)), vector_a, vector_b).to_csv(filename1)
filename2 = "overnight_training30032021\\overnight_training_nme.csv"
# generate_dataset_from_vectors(qt.ket2dm(nme_state(np.pi/8)), vector_a, vector_b).to_csv(filename2)

losses_without_comm_nme = []
histories_without_comm_nme = []
for run in range(3):
    K.clear_session()
    minima, history = train_model_history(filename2)
    losses_without_comm_nme.append(minima)
    histories_without_comm_nme.append(history)
    if minima < config.cutoff:
        break

filename = 'overnight_training30032021\\nme_loss_without_comm.csv'
np.savetxt(filename, losses_without_comm_nme)

count = 0
for run in histories_without_comm_nme:
    filename = 'overnight_training30032021\\history_without_comm_nme_' + str(count+1) + '.csv'
    np.savetxt(filename, histories_without_comm_nme[count])
    count += 1

losses_without_comm_nme = []
histories_without_comm_nme = []

###########################

config.training_size = 48
config.epochs = 1800

vector_a, vector_b = maximum_violation_measurements_extended(np.pi/8, n=48)

np.savetxt('48_vector_list.csv', np.array(np.concatenate((vector_a, vector_b), axis=1)), delimiter=',')

filename_48 = "overnight_training31032021\\overnight_training_48.csv"
generate_dataset_from_vectors(qt.ket2dm(nme_state(np.pi/8)), vector_a, vector_b).to_csv(filename_48)

losses_with_comm_nme = []
histories_with_comm_nme = []
for i in range(8):
    K.clear_session()
    minima, history = train_model_comm_history(filename_48)
    losses_with_comm_nme.append(minima)
    histories_with_comm_nme.append(history)
    if minima < config.cutoff:
        break

filename = 'overnight_training31032021\\nme_loss_with_comm.csv'
np.savetxt(filename, losses_with_comm_nme)

count = 0
for run in histories_with_comm_nme:
    filename = 'overnight_training31032021\\history_with_comm_nme_' + str(count+1) + '.csv'
    np.savetxt(filename, histories_with_comm_nme[count])
    count += 1
