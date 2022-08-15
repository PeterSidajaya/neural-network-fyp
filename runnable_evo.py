import numpy as np
import matplotlib.pyplot as plt
from neural_network_util import *
from evo import Evolution, MP, MP_lin
from evo_protocols import *
from evo_functions import plot_protocol
from evo_fitness import mp_fitness_creator, reference_mp_fitness_creator, mp_lin_fitness_creator
from tensorflow import keras

folder_name = "NewModel\\TV\\pi-16_singlet_1_rev\\"
model = keras.models.load_model(folder_name + "pi_16_model.h5", compile=False)

fitness_creator = mp_lin_fitness_creator(model=model, lhv_size=40, input_size=400, slice='Alice_1')
fitness_creator.generate()
fitness = fitness_creator.create_fitness()

evo = Evolution(
    pool_size=50, fitness=fitness, individual_class=MP_lin, n_offsprings=25,
    pair_params={'alpha': 0.5},
    mutate_params={'std': 0.5, 'dim': 2},
    init_params={'std': 0.5, 'dim': 2},
    param_adjustment=lambda mutate_params, epoch: {'std': mutate_params['std']/(1+0.07*mutate_params['std']), 'dim': mutate_params['dim']} 
)
n_epochs = 20

fitnesses = []
for i in range(n_epochs):
    fitness_creator.generate()
    evo.step()
    print('mutation :', evo.mutate_params['std'])
    # print(evo.pool.individuals[-1].value)
    score = evo.pool.fitness(evo.pool.individuals[-1])
    print('score :', score)
    fitnesses.append(score)


Global_M = evo.pool.individuals[-1].value
    
def custom_protocol(vec_alice, lhv):
    prob = 1/2 - 1/2 * np.sign(mdot(vec_alice, lhv[0] + lhv[1], Global_M[0]))
    return np.array([prob, 1-prob])


def custom_protocol_lin(vec_alice, lhv):
    prob = 1/2 - 1/2 * np.sign(dot(vec_alice, Global_M[0]*lhv[1] + lhv[0] + Global_M[1]*np.array([0,0,1])))
    return np.array([prob, 1-prob])


print(Global_M)
plot_protocol(custom_protocol_lin, (np.array([1,0,0]), np.array([0,0,1])))

Global_M = evo.pool.individuals[-2].value

print(Global_M)
plot_protocol(custom_protocol_lin, (np.array([1,0,0]), np.array([0,0,1])))

plt.plot(fitnesses)
plt.show()