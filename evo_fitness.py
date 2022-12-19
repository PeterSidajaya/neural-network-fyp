from evo_functions import random_vector, random_vectors
import numpy as np
from evo_protocols import *
import config

class mp_fitness_creator():
    def __init__(self, model=None, lhv_size=20, input_size=200, slice='Alice_1') -> None:
        self.model = model
        self.lhv_size = lhv_size
        self.input_size = input_size
        self.slice = slice
    
    def generate(self):
        self.inputs = random_vectors(self.input_size, 3)
        self.lhvs = []
        for i in range(self.lhv_size):
            self.lhvs.append(np.concatenate((random_vector(3), random_vector(3))))
        self.lhvs = np.array(self.lhvs)
        if self.model:
            xarray = np.concatenate((np.tile(self.inputs, (self.lhv_size,1)),np.tile(self.inputs, (self.lhv_size,1)),
                                         np.repeat(self.lhvs, self.input_size, axis=0)), axis=1)
            yarray = self.model.predict(xarray)
            self.yarray = yarray
    
    def create_fitness(self):
        def fitness(M):
            if self.model == None:
                """What did I put this for? I guess we'll never know. I think it's just for testing."""
                pred_probs, true_probs = np.zeros((self.lhv_size*self.input_size, 2)), np.zeros((self.lhv_size*self.input_size, 2))
                for j in range(self.lhv_size):
                    lhv = self.lhvs[j,0:3], self.lhvs[j,3:6]
                    for i in range(self.input_size):
                        input = self.inputs[i]

                        # Edit here for the true and fitted function
                        true_prob = comm_protocol(input, lhv)
                        pred_prob = 1/2 + 1/2 * np.sign(mdot(input, lhv[0], M.value[0])) * np.sign(mdot(input, lhv[1], M.value[0]))
                        
                        true = np.array([true_prob, 1-true_prob])
                        pred = [pred_prob, 1-pred_prob]
                        true_probs[self.input_size*j+i,:] = true
                        pred_probs[self.input_size*j+i,:] = pred

                res = KL_distance(true_probs, pred_probs)/(self.lhv_size*self.input_size)
            
            else:
                probs = np.zeros((self.lhv_size*self.input_size, 2))
                for j in range(self.lhv_size):
                    lhv = self.lhvs[j,0:3], self.lhvs[j,3:6]
                    for i in range(self.input_size):
                        input = self.inputs[i]
                        # Edit here for the true and fitted function
                        prob = 1/2 - 1/2 * np.sign(mdot(input, lhv[0] + lhv[1], M.value[0]))
                        pred = [prob, 1-prob]
                        probs[self.input_size*j+i,:] = pred
                
                yarray = self.yarray
                if self.slice == 'Alice_1':
                    true = yarray[:,1:3]
                elif self.slice == 'Bob_1':
                    true = yarray[:,3:5]
                elif self.slice == 'Alice_2':
                    true = yarray[:,5:7]
                elif self.slice == 'Bob_2':
                    true = yarray[:,7:9]
                elif self.slice == 'Comm':
                    true = np.concatenate((yarray[:,0:1], 1-yarray[:,0:1]), axis=1)
                
                res = KL_distance(true, probs)/(self.lhv_size*self.input_size)
            return -res
        
        return fitness


class mp_lin_fitness_creator():
    def __init__(self, model=None, lhv_size=20, input_size=200, slice='Alice_1') -> None:
        self.model = model
        self.lhv_size = lhv_size
        self.input_size = input_size
        self.slice = slice
    
    def generate(self):
        self.inputs = random_vectors(self.input_size, 3)
        self.lhvs = []
        for i in range(self.lhv_size):
            self.lhvs.append(np.concatenate((random_vector(3), random_vector(3))))
        self.lhvs = np.array(self.lhvs)
        if self.model:
            xarray = np.concatenate((np.tile(self.inputs, (self.lhv_size,1)),np.tile(self.inputs, (self.lhv_size,1)),
                                         np.repeat(self.lhvs, self.input_size, axis=0)), axis=1)
            yarray = self.model.predict(xarray)
            self.yarray = yarray
    
    def create_fitness(self):
        def fitness(M):
            if self.model == None:
                pass
                pred_probs, true_probs = np.zeros((self.lhv_size*self.input_size, 2)), np.zeros((self.lhv_size*self.input_size, 2))
                for j in range(self.lhv_size):
                    lhv = self.lhvs[j,0:3], self.lhvs[j,3:6]
                    for i in range(self.input_size):
                        input = self.inputs[i]

                        # Edit here for the true and fitted function
                        true_prob = comm_protocol(input, lhv)
                        pred_prob = 1/2 + 1/2 * np.sign(mdot(input, lhv[0], M.value[0])) * np.sign(mdot(input, lhv[1], M.value[0]))
                        
                        true = np.array([true_prob, 1-true_prob])
                        pred = [pred_prob, 1-pred_prob]
                        true_probs[self.input_size*j+i,:] = true
                        pred_probs[self.input_size*j+i,:] = pred

                res = KL_distance(true_probs, pred_probs)/(self.lhv_size*self.input_size)
            
            else:
                probs = np.zeros((self.lhv_size*self.input_size, 2))
                for j in range(self.lhv_size):
                    lhv = self.lhvs[j,0:3], self.lhvs[j,3:6]
                    for i in range(self.input_size):
                        input = self.inputs[i]
                        # Edit here for the true and fitted function
                        prob = 1/2 - 1/2 * np.sign(dot(input, M.value[0]*lhv[1] + lhv[0] + M.value[1]*np.array([0,0,1])))
                        pred = [prob, 1-prob]
                        probs[self.input_size*j+i,:] = pred
                
                yarray = self.yarray
                if self.slice == 'Alice_1':
                    true = yarray[:,1:3]
                elif self.slice == 'Bob_1':
                    true = yarray[:,3:5]
                elif self.slice == 'Alice_2':
                    true = yarray[:,5:7]
                elif self.slice == 'Bob_2':
                    true = yarray[:,7:9]
                elif self.slice == 'Comm':
                    true = np.concatenate((yarray[:,0:1], 1-yarray[:,0:1]), axis=1)
                
                res = KL_distance(true, probs)/(self.lhv_size*self.input_size)
            return -res
        
        return fitness


def KL_distance(true, pred):
    eps = np.finfo(np.float64).eps
    pred = np.clip(pred, eps, 1)
    true = np.clip(true, eps, 1)
    return np.sum(true*np.log(true/pred))


def TV_distance(true, pred):
    eps = np.finfo(np.float64).eps
    pred = np.clip(pred, eps, 1)
    true = np.clip(true, eps, 1)
    return 0.5 * np.sum(np.abs(true-pred))


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

class hemisphere_fitness_creator():
    """Class used to generate fitness function to find the hemisphere directions in the outputs of Alice and Bob."""
    def __init__(self, model=None, lhv=(np.array([0,0,1]),np.array([0,0,1])), input_size=1000, slice='Alice_1') -> None:
        self.model = model
        self.lhv = lhv
        self.input_size = input_size
        self.slice = slice
    
    def generate(self):
        self.inputs = random_vectors(self.input_size, 3)
        self.lhvs = np.array([np.concatenate((self.lhv[0], self.lhv[1])),])
        xarray = np.concatenate((self.inputs, self.inputs,
                                        np.repeat(self.lhvs, self.input_size, axis=0)), axis=1)
        self.yarray = self.model.predict(xarray)
    
    def create_fitness(self):
        def fitness(Hemisphere):
            probs = np.zeros((self.input_size, 2))
            for i in range(self.input_size):
                input = self.inputs[i]
                # Edit here for the true and fitted function
                prob = 1/2 - 1/2 * np.sign(dot(input, Hemisphere.value[0]) + Hemisphere.value[1])
                pred = [prob, 1-prob]
                probs[i,:] = pred
            
            if self.slice == 'Alice_1':
                true = self.yarray[:,1:3]
            elif self.slice == 'Bob_1':
                true = self.yarray[:,3:5]
            elif self.slice == 'Alice_2':
                true = self.yarray[:,5:7]
            elif self.slice == 'Bob_2':
                true = self.yarray[:,7:9]
            elif self.slice == 'Comm':
                true = np.concatenate((self.yarray[:,0:1], 1-self.yarray[:,0:1]), axis=1)
            
            res = KL_distance(true, probs)/(self.input_size)
            return -res
        
        return fitness
    
    
class weighting_fitness_creator():
    """Class used to generate fitness function to find the weights in the analytical outpus of Alice and Bob."""
    def __init__(self, model=None, lhv_settings_size=100, input_size=100, slice='Alice_1') -> None:
        self.model = model
        self.lhv_settings_size = lhv_settings_size
        self.input_size = input_size
        self.slice = slice
        config.training_size = 100
        config.LHV_size = 100
    
    def generate(self):
        self.inputs = random_vectors(self.input_size, 3)
        self.inputs_tiled = np.tile(self.inputs, (self.lhv_settings_size, 1))
        
        self.lhv_1 = random_vectors(self.lhv_settings_size, 3)
        self.lhv_2 = random_vectors(self.lhv_settings_size, 3)
        self.lhvs = np.repeat(np.concatenate((self.lhv_1, self.lhv_2), axis=1), self.input_size, axis=0)
        
        xarray = np.concatenate((self.inputs_tiled, self.inputs_tiled, self.lhvs), axis=1)
        self.yarray = self.model.predict(xarray)
    
    def create_fitness(self):
        def fitness(Weighting):
            res = 0
            for i in range(self.lhv_settings_size):
                probs = np.zeros((self.input_size, 2))
                lhv_1, lhv_2 = self.lhv_1[i], self.lhv_2[i]
                for j in range(self.input_size):
                    input = self.inputs[j]
                    # Edit here for the true and fitted function
                    prob = 1/2 - 1/2 * np.sign(dot(input, Weighting.value[0]*lhv_1+lhv_2+Weighting.value[1]*np.array([0,0,1])))
                    pred = [prob, 1-prob]
                    probs[j,:] = pred
            
                if self.slice == 'Alice_1':
                    true = self.yarray[i*self.input_size:(i+1)*self.input_size,1:3]
                elif self.slice == 'Bob_1':
                    true = self.yarray[i*self.input_size:(i+1)*self.input_size,3:5]
                elif self.slice == 'Alice_2':
                    true = self.yarray[i*self.input_size:(i+1)*self.input_size,5:7]
                elif self.slice == 'Bob_2':
                    true = self.yarray[i*self.input_size:(i+1)*self.input_size,7:9]
                res -= KL_distance(true, probs)/(self.input_size)
            return res/self.lhv_settings_size
        
        return fitness
    

class bias_fitness_creator():
    """Class used to find the biases in the analytical outpus of Alice and Bob."""
    def __init__(self, model=None, lhv=(np.array([0,0,1]),np.array([0,0,1])), weights=(1.0,0.0), input_size=2000, slice='Alice_1') -> None:
        self.model = model
        self.lhv = lhv
        self.input_size = input_size
        self.slice = slice
        self.weights = weights
    
    def generate(self):
        self.inputs = random_vectors(self.input_size, 3)
        self.lhvs = np.array([np.concatenate((self.lhv[0], self.lhv[1])),])
        xarray = np.concatenate((self.inputs, self.inputs,
                                        np.repeat(self.lhvs, self.input_size, axis=0)), axis=1)
        self.yarray = self.model.predict(xarray)
    
    def create_fitness(self):
        def fitness(Bias):
            probs = np.zeros((self.input_size, 2))
            for i in range(self.input_size):
                input = self.inputs[i]
                # Edit here for the true and fitted function
                lambda_g = self.weights[0] * self.lhv[0] + self.lhv[1] + self.weights[1] * np.array([0,0,1])
                prob = 1/2 - 1/2 * np.sign(dot(input, lambda_g) + Bias.value)
                pred = [prob, 1-prob]
                probs[i,:] = pred
            
            if self.slice == 'Alice_1':
                true = self.yarray[:,1:3]
            elif self.slice == 'Bob_1':
                true = self.yarray[:,3:5]
            elif self.slice == 'Alice_2':
                true = self.yarray[:,5:7]
            elif self.slice == 'Bob_2':
                true = self.yarray[:,7:9]
            elif self.slice == 'Comm':
                true = np.concatenate((self.yarray[:,0:1], 1-self.yarray[:,0:1]), axis=1)
            
            res = KL_distance(true, probs)/(self.input_size)
            return -res
        
        return fitness