import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from distribution_generator import *
import qutip as qt
import tensorflow.keras as keras
import numpy as np
import config
from model_testing import *


folder_name_1 = "NewModel\\TV_new\\"
folder_name_2 = "NewModel\\prototype\\pi-16_fast_singlet_dot_model_370\\"
folder_name_3 = "NewModel\\SV\\pi-16_singlet_3\\"
state = qt.ket2dm(nme_singlet(np.pi/16))


model_1 = keras.models.load_model(folder_name_1 + "pi_16_model_3.h5", compile=False)
model_2 = keras.models.load_model(folder_name_2 + "pi_16_model.h5", compile=False)
model_3 = keras.models.load_model(folder_name_3 + "pi_16_model.h5", compile=False)
# print(validate(model_2, state, comm=True))

vec_z = np.array([0, 0, 1])
vec_x = np.array([1, 0, 0])
vec_y = np.array([0, 1, 0])
type = 'spherical'
show = False

save_directory = 'NewModel\\prototype\\pi-16_fast_singlet_dot_model_370\\figures\\phi-sweep_7'
os.mkdir(save_directory)
os.mkdir(save_directory + '\\comm')
os.mkdir(save_directory + '\\alice-1')
os.mkdir(save_directory + '\\alice-2')
os.mkdir(save_directory + '\\bob-1')
os.mkdir(save_directory + '\\bob-2')

j = 21
for i in range(j):
    vec_1 = np.cos(i/(j-1) * 2 * np.pi) * vec_x + np.sin(i/(j-1) * 2 * np.pi) * vec_y
    vec_2 =  np.cos(i/(j-1) * 2 * np.pi) * vec_y - np.sin(i/(j-1) * 2 * np.pi) * vec_x
    distr = map_distr_TV(model_2, vec_1, vec_2)
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

