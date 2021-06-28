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

folder_name = "new-LHV\\pi-16_200_SV\\"
state = qt.ket2dm(nme_state(np.pi/16))


config.LHV_type = "vector"
model = keras.models.load_model(folder_name + "pi_16_model.h5", compile=False)
# print(validate(model, state, comm=True))


vec_z = np.array([0, 0, 1])
vec_x = np.array([1, 0, 0])
vec_y = np.array([0, 1, 0])
type = 'spherical'
show = False

save_directory = 'new-LHV\\figures\\SV\\theta-sweep'
os.mkdir(save_directory)
os.mkdir(save_directory + '\\comm')
os.mkdir(save_directory + '\\alice-1')
os.mkdir(save_directory + '\\alice-2')
os.mkdir(save_directory + '\\bob-1')
os.mkdir(save_directory + '\\bob-2')

j = 11
for i in range(j):
    vec = np.cos(i/(j-1) * np.pi) * vec_z + np.sin(i/(j-1) * np.pi) * vec_x
    distr = map_distr_SV(model, vec)
    plot_comm_distr_vector(distr, type=type, color='comm',
                           savename=save_directory+'\\comm\\'+str(i)+'-'+str(j-1)+'comm'+'.png', show=show)
    plot_comm_distr_vector(distr, type=type, color='alice_1',
                           savename=save_directory+'\\alice-1\\'+str(i)+'-'+str(j-1)+'alice-1'+'.png', show=show)
    plot_comm_distr_vector(distr, type=type, color='alice_2',
                           savename=save_directory+'\\alice-2\\'+str(i)+'-'+str(j-1)+'alice-2'+'.png', show=show)
    plot_comm_distr_vector(distr, type=type, color='bob_1',
                           savename=save_directory+'\\bob-1\\'+str(i)+'-'+str(j-1)+'bob-1'+'.png', show=show)
    plot_comm_distr_vector(distr, type=type, color='bob_2',
                           savename=save_directory+'\\bob-2\\'+str(i)+'-'+str(j-1)+'bob-2'+'.png', show=show)

