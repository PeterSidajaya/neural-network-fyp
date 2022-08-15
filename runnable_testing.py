import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from distribution_generator import *
import qutip as qt
import tensorflow.keras as keras
import numpy as np
import config
from model_testing import *
from auxiliary import *
from matplotlib import rcParams
rcParams['font.family'] = 'serif'


# qutrit
folder_name = "qutrit\\"
state = qt.tensor(qt.identity(3), qt.identity(3))


model = keras.models.load_model(folder_name + "qubit.h5", compile=False)
config.training_size = 1
print(predict(model, np.array([[0,0,1,0,0,1]]), comm=False))

# old
folder_name = "NewModel\\TV\\3pi-16_singlet_2\\"
theta = 3*np.pi/16
state = qt.ket2dm(nme_singlet(theta))


model = keras.models.load_model(folder_name + "3pi_16_model.h5", compile=False)
config.training_size = 1
print(validate(model, state, comm=True, type='tvd'))

vec_alice = random_unit_vector(3)
vec_bob = random_unit_vector(3)

print(comm_balance(model, vec_alice, vec_bob))

print('STRATEGY ONE')
evaluate_marginals(model, theta, vec_alice, vec_bob, local=True, strategy=0)

print('STRATEGY TWO')
evaluate_marginals(model, theta, vec_alice, vec_bob, local=True, strategy=1)

# vec_x = np.array([1, 0, 0])
# vec_y = np.array([0, 1, 0])
# vec_z = np.array([0, 0, 1])
# base = {'x': vec_x,
#         'y': vec_y,
#         'z': vec_z}

# type = 'scatter'
# show = False

# distr = map_distr_TV(model, vec_z, vec_z)
# plot_comm_distr_vector(distr, type=type, color='comm', show=True, title="Bit of communication sent")

# rt = folder_name + "figures3D"
# os.mkdir(rt)

# for l_sweep, l_fix in [(1, 2), (2, 1)]:
#     root = rt + '\\lhv_{}'.format(l_sweep)
#     os.mkdir(root)
#     os.mkdir(root + '\\theta_sweep')
#     os.mkdir(root + '\\phi_sweep')
    
#     for d in ['x', 'y', 'z']:
#         parent = root + '\\theta_sweep' + '\\{}-{}'.format(l_fix, d)
#         os.mkdir(parent)
#         os.mkdir(parent + '\\comm')
#         os.mkdir(parent + '\\alice-1')
#         os.mkdir(parent + '\\alice-2')
#         os.mkdir(parent + '\\bob-1')
#         os.mkdir(parent + '\\bob-2')

#         j = 31
#         for i in range(j):
#             if l_sweep == 1:
#                 vec_1 = np.cos(i/(j-1) * np.pi) * vec_z + np.sin(i/(j-1) * np.pi) * vec_x
#                 vec_2 = base[d]
#             else:
#                 vec_2 = np.cos(i/(j-1) * np.pi) * vec_z + np.sin(i/(j-1) * np.pi) * vec_x
#                 vec_1 = base[d]
            
#             distr = map_distr_TV(model, vec_1, vec_2)
#             plot_comm_distr_vector(distr, type=type, color='comm',
#                                 savename=parent+'\\comm\\'+str(i)+'-'+str(j-1)+'comm'+'.png', show=show, title="Bit of communication sent")
#             plot_comm_distr_vector(distr, type=type, color='alice_1',
#                                 savename=parent+'\\alice-1\\'+str(i)+'-'+str(j-1)+'alice-1'+'.png', show=show, title="Alice's first output")
#             plot_comm_distr_vector(distr, type=type, color='alice_2',
#                                 savename=parent+'\\alice-2\\'+str(i)+'-'+str(j-1)+'alice-2'+'.png', show=show, title="Alice's second output")
#             plot_comm_distr_vector(distr, type=type, color='bob_1',
#                                 savename=parent+'\\bob-1\\'+str(i)+'-'+str(j-1)+'bob-1'+'.png', show=show, title="Bob's first output")
#             plot_comm_distr_vector(distr, type=type, color='bob_2',
#                                 savename=parent+'\\bob-2\\'+str(i)+'-'+str(j-1)+'bob-2'+'.png', show=show, title="Bob's second output")
#             plt.close('all')
#         make_gif(parent)

#         parent = root + '\\phi_sweep' + '\\{}-{}'.format(l_fix, d)
#         os.mkdir(parent)
#         os.mkdir(parent + '\\comm')
#         os.mkdir(parent + '\\alice-1')
#         os.mkdir(parent + '\\alice-2')
#         os.mkdir(parent + '\\bob-1')
#         os.mkdir(parent + '\\bob-2')

#         j = 31
#         for i in range(j):
#             if l_sweep == 1:
#                 vec_1 = np.cos(i/(j-1) * 2 * np.pi) * vec_x + np.sin(i/(j-1) * 2 * np.pi) * vec_y
#                 vec_2 = base[d]
#             else:
#                 vec_2 = np.cos(i/(j-1) * 2 * np.pi) * vec_x + np.sin(i/(j-1) * 2 * np.pi) * vec_y
#                 vec_1 = base[d]
            
#             distr = map_distr_TV(model, vec_1, vec_2)
#             plot_comm_distr_vector(distr, type=type, color='comm',
#                                 savename=parent+'\\comm\\'+str(i)+'-'+str(j-1)+'comm'+'.png', show=show, title="Bit of communication sent")
#             plot_comm_distr_vector(distr, type=type, color='alice_1',
#                                 savename=parent+'\\alice-1\\'+str(i)+'-'+str(j-1)+'alice-1'+'.png', show=show, title="Alice's first output")
#             plot_comm_distr_vector(distr, type=type, color='alice_2',
#                                 savename=parent+'\\alice-2\\'+str(i)+'-'+str(j-1)+'alice-2'+'.png', show=show, title="Alice's second output")
#             plot_comm_distr_vector(distr, type=type, color='bob_1',
#                                 savename=parent+'\\bob-1\\'+str(i)+'-'+str(j-1)+'bob-1'+'.png', show=show, title="Bob's first output")
#             plot_comm_distr_vector(distr, type=type, color='bob_2',
#                                 savename=parent+'\\bob-2\\'+str(i)+'-'+str(j-1)+'bob-2'+'.png', show=show, title="Bob's second output")
#             plt.close('all')
#         make_gif(parent)
