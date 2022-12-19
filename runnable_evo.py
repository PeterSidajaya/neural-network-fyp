import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import neural_network_util
from evo import Evolution, Hemisphere, Weighting, Bias
from evo_functions import plot_protocol
from evo_fitness import hemisphere_fitness_creator, weighting_fitness_creator, bias_fitness_creator
from tensorflow import keras
from auxiliary import *

folder_name = "NewModel\\TV\\pi-8_singlet_1\\"
model = keras.models.load_model(folder_name + "pi_8_model.h5", compile=False)

lhv_1 = np.array([1,0,0])
lhv_2 = np.array([0,1,0])

def find_best_weight(model, slice='Alice_1', epoch=100):
    fitness_creator = weighting_fitness_creator(model=model, lhv_settings_size=20, input_size=500, slice=slice)
    fitness_creator.generate()
    fitness = fitness_creator.create_fitness()

    evo = Evolution(
        pool_size=50, fitness=fitness, individual_class=Weighting, n_offsprings=20,
        pair_params={'alpha': 0.5},
        mutate_params={'std': 0.1},
        init_params={'loc_weight': 0.0, 'loc_drag': 0.0, 'std': 0.5},
        # param_adjustment=lambda mutate_params, epoch: {'std': mutate_params['std']/(1+0.05*mutate_params['std'])} 
    )
    n_epochs = epoch

    for i in range(n_epochs):
        fitness_creator.generate()
        print('epoch :', i)
        evo.step()
        # print('mutation :', evo.mutate_params['std'])
        score = evo.pool.fitness(evo.pool.individuals[-1])
        print(evo.pool.individuals[-1].value)
        print(score)

    return evo.pool.individuals


def find_best_bias(model, lhv=(np.array([1,0,0]),np.array([0,0,1])), weights=(1.42977042, -0.7230522), slice='Alice_1', epoch=10):
    fitness_creator = bias_fitness_creator(model=model, lhv=lhv, weights=weights, slice=slice)
    fitness_creator.generate()
    fitness = fitness_creator.create_fitness()

    evo = Evolution(
        pool_size=50, fitness=fitness, individual_class=Bias, n_offsprings=20,
        pair_params={'alpha': 0.5},
        mutate_params={'std': 0.05},
        init_params={'loc': -1.0, 'std': 0.5},
        # param_adjustment=lambda mutate_params, epoch: {'std': mutate_params['std']/(1+0.05*mutate_params['std'])} 
    )
    n_epochs = epoch

    for i in range(n_epochs):
        fitness_creator.generate()
        # print('epoch :', i)
        evo.step()
        # print('mutation :', evo.mutate_params['std'])
        score = evo.pool.fitness(evo.pool.individuals[-1])
        # print(evo.pool.individuals[-1].value)
    print(score)

    return evo.pool.individuals


def find_best_bias_brute(model, lhv=(np.array([1,0,0]),np.array([1,0,0])), weights=(1.42977042, -0.7230522), slice='Alice_1', epoch=10):
    fitness_creator = bias_fitness_creator(model=model, lhv=lhv, weights=weights, slice=slice, input_size=5000)
    fitness_creator.generate()
    fitness = fitness_creator.create_fitness()

    biases = np.linspace(2,-2,250)
    max_bias = 0
    max_score = -9999

    for bias in biases:
        score = fitness(Bias(bias))
        if score > max_score:
            max_score = score
            max_bias = bias
            
    return max_bias, max_score

# best = find_best_weight(model)
# array = np.array([0.0,0.0])
# for i in range(10):
#     array += np.array(best[-i].value) / 10
# print(array)

# best = find_best_bias(model)
# print(best[-1].value)


theta_array,phi_array = np.meshgrid(np.linspace(0,np.pi,20), np.linspace(0,2*np.pi,21))
fig, ax = plt.subplots()
c = np.ndarray((21,20))
for i in range(21):
    for j in range(20):
        theta = theta_array[i,j]
        phi = phi_array[i,j]
        lhv_var = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        c[i,j] = find_best_bias_brute(model, lhv=(lhv_var, np.array([1,0,0])))[0]
ax.pcolormesh(phi_array, theta_array, c)
np.savetxt('NewModel\\TV\\pi-8_singlet_1\\evo_guess\\bias_guess\\' + 'phi.txt', phi_array)
np.savetxt('NewModel\\TV\\pi-8_singlet_1\\evo_guess\\bias_guess\\' + 'theta.txt', theta_array)
np.savetxt('NewModel\\TV\\pi-8_singlet_1\\evo_guess\\bias_guess\\' + 'bias.txt', c)

plt.show()
