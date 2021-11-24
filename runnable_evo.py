import numpy as np
import matplotlib.pyplot as plt
from neural_network_util import *
from evo import Evolution, MP
from evo_protocols import *
from evo_functions import plot_protocol
from evo_fitness import mp_fitness_creator, reference_mp_fitness_creator
from tensorflow import keras

folder_name = "NewModel\\TV\\pi-4_singlet_1_rev\\"
model = keras.models.load_model(folder_name + "pi_4_model.h5", compile=False)

fitness = mp_fitness_creator(model=model, lhv_size=15, input_size=150, slice='Comm')

evo = Evolution(
    pool_size=120, fitness=fitness, individual_class=MP, n_offsprings=60,
    pair_params={'alpha': 0.5},
    mutate_params={'std': 0.5, 'dim': 1},
    init_params={'std': 1, 'dim': 1},
    param_adjustment=lambda mutate_params, epoch: {'std': mutate_params['std'] * 150/(150+epoch), 'dim': mutate_params['dim']} 
)
n_epochs = 50

fitnesses = []
for i in range(n_epochs):
    evo.step()
    print('mutation :', evo.mutate_params['std'])
    print(evo.pool.individuals[-1].value)
    score = evo.pool.fitness(evo.pool.individuals[-1])
    print('score :', score)
    fitnesses.append(score)


Global_M = evo.pool.individuals[-1].value
    
def custom_protocol(vec_alice, lhv):
    prob = 1/2 + 1/2 * np.sign(mdot(vec_alice, lhv[0], Global_M[0])) * np.sign(mdot(vec_alice, lhv[1], Global_M[0]))
    return prob

print(Global_M)
plot_protocol(custom_protocol, (np.array([1,0,0]), np.array([0,0,1])))
plot_protocol(comm_protocol, (np.array([1,0,0]), np.array([0,0,1])))
plt.plot(fitnesses)
plt.show()