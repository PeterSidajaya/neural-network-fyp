from grapher import *
from training import *
from distribution_generator import *
import pickle
import qutip as qt
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import config
from matplotlib import cm
from neural_network_util import build_model_comm, comm_customLoss_multiple
from model_testing import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


folder_name = "symmetry\\pi-16_250_SV_singlet\\"
state = qt.ket2dm(nme_singlet(np.pi/16))


model = keras.models.load_model(folder_name + "pi_16_model.h5", compile=False)
# print(validate(model, state, comm=True))

evaluate_marginals(model, np.pi/16, random_unit_vector(3),
                   random_unit_vector(3), singlet=True)

plot_marginal_alice_semicircle(model, fix=True)


vec_z = np.array([0, 1])
vec_x = np.array([1, 0])
type = 'spherical'
show = False

"""
save_directory = 'symmetry\\figures\\SV-singlet\\theta-sweep-comm-detailed'
os.mkdir(save_directory)
os.mkdir(save_directory + '\\comm')
os.mkdir(save_directory + '\\comm-detailed')
os.mkdir(save_directory + '\\alice-1')
os.mkdir(save_directory + '\\alice-2')
os.mkdir(save_directory + '\\bob-1')
os.mkdir(save_directory + '\\bob-2')
"""

j = 51
max = []
min = []
avg = []
for i in range(j):
    vec = np.cos(i/(j-1) * np.pi) * vec_z + np.sin(i/(j-1) * np.pi) * vec_x
    distr = map_distr(model, vec, type='semicircle')
    comm_list = distr.c
    max.append(comm_list.max())
    min.append(comm_list.min())
    avg.append(comm_list.mean())
    """
    plot_comm_distr_vector(distr, type=type, color='comm',
                           savename=save_directory+'\\comm\\'+str(i)+'-'+str(j-1)+'comm'+'.png', show=show)
    plot_comm_distr_vector(distr, type=type, color='comm',
                           savename=save_directory+'\\comm-detailed\\'+str(i)+'-'+str(j-1)+'comm'+'.png', show=show, fix_color=False)
    plot_comm_distr_vector(distr, type=type, color='alice_1',
                           savename=save_directory+'\\alice-1\\'+str(i)+'-'+str(j-1)+'alice-1'+'.png', show=show)
    plot_comm_distr_vector(distr, type=type, color='alice_2',
                           savename=save_directory+'\\alice-2\\'+str(i)+'-'+str(j-1)+'alice-2'+'.png', show=show)
    plot_comm_distr_vector(distr, type=type, color='bob_1',
                           savename=save_directory+'\\bob-1\\'+str(i)+'-'+str(j-1)+'bob-1'+'.png', show=show)
    plot_comm_distr_vector(distr, type=type, color='bob_2',
                           savename=save_directory+'\\bob-2\\'+str(i)+'-'+str(j-1)+'bob-2'+'.png', show=show)
    """

def high_low():
    A = -0.498926737578
    B = 0.065625
    a = np.pi/6
    b = np.pi/2
    c = 5 * np.pi/6
    p = np.polynomial.Polynomial([B,-A*a*b*c,A/2*(a*b+b*c+c*a),-A*1/3*(a+b+c),A/4])
    return p

xi = np.linspace(0, np.pi, 51)
plt.plot(xi, max, label='model max')
plt.plot(xi, min, label='model min')
plt.plot(xi, avg, label='model mean')
poly = high_low()
plt.plot(xi, poly(xi)+0.15, label='approx max')
plt.plot(xi, poly(xi)-0.15, label='approx min')
plt.legend()
plt.show()
