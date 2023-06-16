import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import matplotlib.pyplot as plt
import neural_network_util
from evo import Evolution, Hemisphere, Weighting, Bias, Bias_coefficients, Comm_coefficients
from evo_functions import plot_protocol
from evo_fitness import hemisphere_fitness_creator, weighting_fitness_creator, bias_fitness_creator, bias_coefficient_fitness_creator, comm_coefficient_fitness_creator
from tensorflow import keras
from auxiliary import *

"""
This file is the runnable that is run to find the parameters of the semianalytical protocol.
Now, this is a very long file. I will try to guide the interested reader through the file using the comments.
"""

# Pick the model 
folder_name = "qubits-simple/qubit-pi-8-simple/"
model = keras.models.load_model(folder_name + "pi-8_model.h5", compile=False)

lhv_1 = np.array([1,0,0])
lhv_2 = np.array([0,1,0])

def find_best_weight(model, slice='Alice_1', epoch=100, sign=-1):
    """This will be used to find the weights of the hemisphere normal vector. They are the u and v in the semianalytical protocol."""
    fitness_creator = weighting_fitness_creator(model=model, lhv_settings_size=20, input_size=500, slice=slice, sign=sign)
    fitness_creator.generate()
    sys.stdout.write('\033[2K\033[1G')
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
        sys.stdout.write('\033[2K\033[1G')
        # print('epoch :', i)
        evo.step()
        # print('mutation :', evo.mutate_params['std'])
        score = evo.pool.fitness(evo.pool.individuals[-1])
        # print(evo.pool.individuals[-1].value)
        # print(score)

    return evo.pool.individuals


def find_best_bias(model, lhv=(np.array([1,0,0]),np.array([0,0,1])), weights=(1.42977042, -0.7230522), slice='Alice_1', epoch=10, sign=-1):
    """This is the function that is used to find the bias for a specific lhv setting, the b_a1, b_a2 and so forth in the paper."""
    fitness_creator = bias_fitness_creator(model=model, lhv=lhv, weights=weights, slice=slice, sign=sign)
    fitness_creator.generate()
    sys.stdout.write('\033[2K\033[1G')
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
        sys.stdout.write('\033[2K\033[1G')
        # print('epoch :', i)
        evo.step()
        # print('mutation :', evo.mutate_params['std'])
        score = evo.pool.fitness(evo.pool.individuals[-1])
        # print(evo.pool.individuals[-1].value)
    # print(score)

    return evo.pool.individuals


def find_best_bias_brute(model, lhv=(np.array([1,0,0]),np.array([1,0,0])), weights=(1.42977042, -0.7230522), slice='Alice_1', sign=-1):
    """Same function as find_best_bias, but we realised that a brute force approach is more efficient."""
    fitness_creator = bias_fitness_creator(model=model, lhv=lhv, weights=weights, slice=slice, input_size=500, sign=sign)
    fitness_creator.generate()
    sys.stdout.write('\033[2K\033[1G')
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

def find_bias_coefficient(epoch=100, folder='bias-guess/Alice-1/'):
    """Used to do a regression to find the value of w, x, and y of the bias function.

    Technically we can use normal regression, but we did a single test and found that the evolutionary algorithm
    gives a good enough result compared to a regression, esp. considering that this method can be used for nonlinear
    regression of any weird functions, though we did not use that flexibility in the end.
    """
    fitness_creator = bias_coefficient_fitness_creator(folder)
    fitness = fitness_creator.create_fitness()

    evo = Evolution(
        pool_size=100, fitness=fitness, individual_class=Bias_coefficients, n_offsprings=50,
        pair_params={'alpha': 0.5},
        mutate_params={'std': 0.1},
        init_params={'loc': 0.0, 'std': 0.5},
    )
    n_epochs = epoch

    for i in range(n_epochs):
        # print('epoch :', i)
        evo.step()
        score = evo.pool.fitness(evo.pool.individuals[-1])
        # print(score)
        # print(evo.pool.individuals[-1].value)

    return evo.pool.individuals

def find_comm_coefficients(model, epoch=50, lhv_settings_size=100, input_size=500):
    """Used to find the coefficients of the communication model using straightforward evolutionary algorithm."""
    fitness_creator = comm_coefficient_fitness_creator(model=model, lhv_settings_size=lhv_settings_size, input_size=input_size)
    fitness_creator.generate()
    sys.stdout.write('\033[2K\033[1G')
    fitness = fitness_creator.create_fitness()

    evo = Evolution(
        pool_size=40, fitness=fitness, individual_class=Comm_coefficients, n_offsprings=20,
        pair_params={'alpha': 0.5},
        mutate_params={'std': 0.1},
        init_params={'loc': 0.5, 'std': 0.5},
        # param_adjustment=lambda mutate_params, epoch: {'std': mutate_params['std']/(1+0.05*mutate_params['std'])} 
    )
    n_epochs = epoch

    for i in range(n_epochs):
        if i % 2 == 0:
            fitness_creator.generate()
            sys.stdout.write('\033[2K\033[1G')
        # print('epoch :', i)
        evo.step()
        # print('mutation :', evo.mutate_params['std'])
        score = evo.pool.fitness(evo.pool.individuals[-1])
        # print(evo.pool.individuals[-1].value)
        # print(score)

    return evo.pool.individuals

print("Initiating Parameter Search")
print("Current model: ", folder_name[:-1])
print("Finding vector weights")

"""First, we find the weights of the hemisphere vector, the u and v."""

best = find_best_weight(model, slice='Alice_1', sign=-1)
array = np.array([0.0,0.0])
for i in range(5):
    array += np.array(best[-i].value) / 5
weights_alice_1 = array

best = find_best_weight(model, slice='Alice_2', sign=+1)
array = np.array([0.0,0.0])
for i in range(5):
    array += np.array(best[-i].value) / 5
weights_alice_2 = array

best = find_best_weight(model, slice='Bob_1', sign=+1)
array = np.array([0.0,0.0])
for i in range(5):
    array += np.array(best[-i].value) / 5
weights_bob_1 = array

best = find_best_weight(model, slice='Bob_2', sign=-1)
array = np.array([0.0,0.0])
for i in range(5):
    array += np.array(best[-i].value) / 5
weights_bob_2 = array

print("Vector weights")
print("Alice 1: ", weights_alice_1)
print("Alice 2: ", weights_alice_2)
print("Bob 1  : ", weights_bob_1)
print("Bob 2  : ", weights_bob_2)

"""
weights_alice_1 = (-0.6095, -0.035)
weights_alice_2 = (0.6150, 0.7418)
weights_bob_1 = (-0.6872, 0.0473)
weights_bob_2 = (0.5261, 0.5859)
"""

"""Then, we find the numerical biases at some lhv_settings."""

def find_biases(subfolder_name, weights, slice, sign):
    os.mkdir(folder_name + subfolder_name)
    os.mkdir(folder_name + subfolder_name + '2-z')
    os.mkdir(folder_name + subfolder_name + '2-x')
    os.mkdir(folder_name + subfolder_name + '2--z')
    os.mkdir(folder_name + subfolder_name + '1-z')
    os.mkdir(folder_name + subfolder_name + '1-x')
    os.mkdir(folder_name + subfolder_name + '1--z')

    theta_array,phi_array = np.meshgrid(np.linspace(0,np.pi,20), np.linspace(0,2*np.pi,21))
    fig, ax = plt.subplots()
    c = np.ndarray((21,20))
    for i in range(21):
        for j in range(20):
            theta = theta_array[i,j]
            phi = phi_array[i,j]
            lhv_var = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
            c[i,j] = find_best_bias_brute(model, lhv=(lhv_var, np.array([0,0,1])), weights=weights, slice=slice, sign=sign)[0]
    ax.pcolormesh(phi_array, theta_array, c)
    np.savetxt(folder_name + subfolder_name + '2-z/' + 'phi.txt', phi_array)
    np.savetxt(folder_name + subfolder_name + '2-z/' + 'theta.txt', theta_array)
    np.savetxt(folder_name + subfolder_name + '2-z/' + 'bias.txt', c)

    theta_array,phi_array = np.meshgrid(np.linspace(0,np.pi,20), np.linspace(0,2*np.pi,21))
    fig, ax = plt.subplots()
    c = np.ndarray((21,20))
    for i in range(21):
        for j in range(20):
            theta = theta_array[i,j]
            phi = phi_array[i,j]
            lhv_var = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
            c[i,j] = find_best_bias_brute(model, lhv=(lhv_var, np.array([1,0,0])), weights=weights, slice=slice, sign=sign)[0]
    ax.pcolormesh(phi_array, theta_array, c)
    np.savetxt(folder_name + subfolder_name + '2-x/' + 'phi.txt', phi_array)
    np.savetxt(folder_name + subfolder_name + '2-x/' + 'theta.txt', theta_array)
    np.savetxt(folder_name + subfolder_name + '2-x/' + 'bias.txt', c)

    theta_array,phi_array = np.meshgrid(np.linspace(0,np.pi,20), np.linspace(0,2*np.pi,21))
    fig, ax = plt.subplots()
    c = np.ndarray((21,20))
    for i in range(21):
        for j in range(20):
            theta = theta_array[i,j]
            phi = phi_array[i,j]
            lhv_var = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
            c[i,j] = find_best_bias_brute(model, lhv=(lhv_var, np.array([0,0,-1])), weights=weights, slice=slice, sign=sign)[0]
    ax.pcolormesh(phi_array, theta_array, c)
    np.savetxt(folder_name + subfolder_name + '2--z/' + 'phi.txt', phi_array)
    np.savetxt(folder_name + subfolder_name + '2--z/' + 'theta.txt', theta_array)
    np.savetxt(folder_name + subfolder_name + '2--z/' + 'bias.txt', c)

    theta_array,phi_array = np.meshgrid(np.linspace(0,np.pi,20), np.linspace(0,2*np.pi,21))
    fig, ax = plt.subplots()
    c = np.ndarray((21,20))
    for i in range(21):
        for j in range(20):
            theta = theta_array[i,j]
            phi = phi_array[i,j]
            lhv_var = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
            c[i,j] = find_best_bias_brute(model, lhv=(np.array([0,0,1]), lhv_var), weights=weights, slice=slice, sign=sign)[0]
    ax.pcolormesh(phi_array, theta_array, c)
    np.savetxt(folder_name + subfolder_name + '1-z/' + 'phi.txt', phi_array)
    np.savetxt(folder_name + subfolder_name + '1-z/' + 'theta.txt', theta_array)
    np.savetxt(folder_name + subfolder_name + '1-z/' + 'bias.txt', c)

    theta_array,phi_array = np.meshgrid(np.linspace(0,np.pi,20), np.linspace(0,2*np.pi,21))
    fig, ax = plt.subplots()
    c = np.ndarray((21,20))
    for i in range(21):
        for j in range(20):
            theta = theta_array[i,j]
            phi = phi_array[i,j]
            lhv_var = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
            c[i,j] = find_best_bias_brute(model, lhv=(np.array([1,0,0]), lhv_var), weights=weights, slice=slice, sign=sign)[0]
    ax.pcolormesh(phi_array, theta_array, c)
    np.savetxt(folder_name + subfolder_name + '1-x/' + 'phi.txt', phi_array)
    np.savetxt(folder_name + subfolder_name + '1-x/' + 'theta.txt', theta_array)
    np.savetxt(folder_name + subfolder_name + '1-x/' + 'bias.txt', c)

    theta_array,phi_array = np.meshgrid(np.linspace(0,np.pi,20), np.linspace(0,2*np.pi,21))
    fig, ax = plt.subplots()
    c = np.ndarray((21,20))
    for i in range(21):
        for j in range(20):
            theta = theta_array[i,j]
            phi = phi_array[i,j]
            lhv_var = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
            c[i,j] = find_best_bias_brute(model, lhv=(np.array([0,0,-1]), lhv_var), weights=weights, slice=slice, sign=sign)[0]
    ax.pcolormesh(phi_array, theta_array, c)
    np.savetxt(folder_name + subfolder_name + '1--z/' + 'phi.txt', phi_array)
    np.savetxt(folder_name + subfolder_name + '1--z/' + 'theta.txt', theta_array)
    np.savetxt(folder_name + subfolder_name + '1--z/' + 'bias.txt', c)

os.mkdir(folder_name + 'bias-guess')
os.mkdir(folder_name + 'bias-guess/Alice_1')
os.mkdir(folder_name + 'bias-guess/Alice_2')
os.mkdir(folder_name + 'bias-guess/Bob_1')
os.mkdir(folder_name + 'bias-guess/Bob_2')

"""Then, we find do a regression on those numerical biases to get w, x, and y."""

print("Finding biases")
find_biases('bias-guess/Alice-1/',weights_alice_1,'Alice_1',-1)
find_biases('bias-guess/Alice-2/',weights_alice_2,'Alice_2',+1)
find_biases('bias-guess/Bob-1/',weights_bob_1,'Bob_1',+1)
find_biases('bias-guess/Bob-2/',weights_bob_2,'Bob_2',-1)

print("Finding bias coefficients")
Coeff_alice_1 = find_bias_coefficient(folder=folder_name + 'bias-guess/Alice-1/')
Coeff_alice_2 = find_bias_coefficient(folder=folder_name + 'bias-guess/Alice-2/')
Coeff_bob_1 = find_bias_coefficient(folder=folder_name + 'bias-guess/Bob-1/')
Coeff_bob_2 = find_bias_coefficient(folder=folder_name + 'bias-guess/Bob-2/')

print("Bias coefficients:")
print("Alice 1: ", Coeff_alice_1[-1].value)
print("Alice 2: ", Coeff_alice_2[-1].value)
print("Bob 1  : ", Coeff_bob_1[-1].value)
print("Bob 2  : ", Coeff_bob_2[-1].value)

"""Then, we find the parameters of the communication functions."""

print("Finding comm coefficients")
Coeff_comm = find_comm_coefficients(model=model)
print("Comm coefficient: ", Coeff_comm[-1].value)

print("Vector weights")
print("Alice 1: ", weights_alice_1)
print("Alice 2: ", weights_alice_2)
print("Bob 1  : ", weights_bob_1)
print("Bob 2  : ", weights_bob_2)

print("Bias coefficients:")
print("Alice 1: ", Coeff_alice_1[-1].value)
print("Alice 2: ", Coeff_alice_2[-1].value)
print("Bob 1  : ", Coeff_bob_1[-1].value)
print("Bob 2  : ", Coeff_bob_2[-1].value)

print("Comm coefficient: ", Coeff_comm[-1].value)


