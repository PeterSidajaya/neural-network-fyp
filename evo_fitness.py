from evo_functions import random_vector, random_vectors
import numpy as np
from evo_protocols import *

def mp_fitness_creator(model=None, lhv_size=20, input_size=200, slice='Alice_1'):
    
    def fitness(M):
        inputs = random_vectors(input_size, 3)
        lhvs = []
        for i in range(lhv_size):
            lhvs.append(np.concatenate((random_vector(3), random_vector(3))))
        lhvs = np.array(lhvs)
        
        if model == None:
            pred_probs, true_probs = np.zeros((lhv_size*input_size, 2)), np.zeros((lhv_size*input_size, 2))
            for j in range(lhv_size):
                lhv = lhvs[j,0:3], lhvs[j,3:6]
                for i in range(input_size):
                    input = inputs[i]

                    # Edit here for the true and fitted function
                    true_prob = comm_protocol(input, lhv)
                    pred_prob = 1/2 + 1/2 * np.sign(mdot(input, lhv[0], M.value[0])) * np.sign(mdot(input, lhv[1], M.value[0]))
                    
                    true = np.array([true_prob, 1-true_prob])
                    pred = [pred_prob, 1-pred_prob]
                    true_probs[input_size*j+i,:] = true
                    pred_probs[input_size*j+i,:] = pred

            res = KL_distance(true_probs, pred_probs)/(lhv_size*input_size)
        
        else:
            xarray = np.concatenate((np.tile(inputs, (lhv_size,1)), np.tile(inputs, (lhv_size,1)), np.repeat(lhvs, input_size, axis=0)), axis=1)
            yarray = model.predict(xarray)
            probs = np.zeros((lhv_size*input_size, 2))
            for j in range(lhv_size):
                lhv = lhvs[j,0:3], lhvs[j,3:6]
                for i in range(input_size):
                    input = inputs[i]
                    # Edit here for the true and fitted function
                    prob = 1/2 + 1/2 * np.sign(mdot(input, lhv[0], M.value[0])) * np.sign(mdot(input, lhv[1], M.value[0]))
                    pred = [prob, 1-prob]
                    probs[input_size*j+i,:] = pred
            if slice == 'Alice_1':
                true = yarray[:,1:3]
            elif slice == 'Bob_1':
                true = yarray[:,3:5]
            elif slice == 'Alice_2':
                true = yarray[:,5:7]
            elif slice == 'Bob_2':
                true = yarray[:,7:9]
            elif slice == 'Comm':
                true = np.concatenate((yarray[:,0:1], 1-yarray[:,0:1]), axis=1)
            res = KL_distance(true, probs)/(lhv_size*input_size)
        return -res

    return fitness


def KL_distance(true, pred):
    eps = np.finfo(np.float64).eps
    pred = np.clip(pred, eps, 1)
    true = np.clip(true, eps, 1)
    return np.sum(true*np.log(true/pred))


def reference_mp_fitness_creator(lhv_size=20, input_size=200):
    eps = np.finfo(np.float64).eps

    def fitness(M):
        lhvs = []
        for i in range(lhv_size):
            lhvs.append((random_vector(3),random_vector(3)))
    
        inputs = []
        for i in range(input_size):
            inputs.append(random_vector(3))
        
        res = 0
        for j in range(lhv_size):
            lhv = lhvs[j]
            for i in range(input_size):
                input = inputs[i]
                
                prob = comm_protocol(input, lhv)
                true = [prob, 1-prob]
                prob = 1/2 + 1/4 * np.sign(mdot(input, lhv[0], M.value[0])) * np.sign(mdot(input, lhv[1], M.value[0])) \
                        + 1/4 * np.sign(mdot(input, lhv[0], M.value[1])) * np.sign(mdot(input, lhv[1], M.value[1]))
                pred = [prob, 1-prob]
                
                pred = np.clip(pred, eps, 1)
                true = np.clip(true, eps, 1)
                res += np.sum(true*np.log(true/pred))/(lhv_size*input_size)
        return -res

    return fitness