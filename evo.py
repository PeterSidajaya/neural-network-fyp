import numpy as np
from abc import ABC, abstractmethod

"""Our genetic evolutionary algorithm is taken from
https://towardsdatascience.com/an-extensible-evolutionary-algorithm-example-in-python-7372c56a557b

We just extended it for our purposes.
"""


class Individual(ABC):
    def __init__(self, value=None, init_params=None):
        if value is not None:
            self.value = value
        else:
            self.value = self._random_init(init_params)

    @abstractmethod
    def pair(self, other, pair_params):
        pass

    @abstractmethod
    def mutate(self, mutate_params):
        pass

    @abstractmethod
    def _random_init(self, init_params):
        pass


class Optimization(Individual):
    def pair(self, other, pair_params):
        return Optimization(pair_params['alpha'] * self.value + (1 - pair_params['alpha']) * other.value)

    def mutate(self, mutate_params):
        self.value += np.random.normal(0, mutate_params['rate'], mutate_params['dim'])
        for i in range(len(self.value)):
            if self.value[i] < mutate_params['lower_bound']:
                self.value[i] = mutate_params['lower_bound']
            elif self.value[i] > mutate_params['upper_bound']:
                self.value[i] = mutate_params['upper_bound']

    def _random_init(self, init_params):
        return np.random.uniform(init_params['lower_bound'], init_params['upper_bound'], init_params['dim'])


class Population:
    def __init__(self, size, fitness, individual_class, init_params):
        self.fitness = fitness
        self.individuals = [individual_class(init_params=init_params) for _ in range(size)]
        self.individuals.sort(key=lambda x: self.fitness(x))

    def replace(self, new_individuals):
        size = len(self.individuals)
        self.individuals.extend(new_individuals)
        self.individuals.sort(key=lambda x: self.fitness(x))
        self.individuals = self.individuals[-size:]

    def get_parents(self, n_offsprings):
        mothers = self.individuals[-2 * n_offsprings::2]
        fathers = self.individuals[-2 * n_offsprings + 1::2]

        return mothers, fathers


class Evolution:
    def __init__(self, pool_size, fitness, individual_class, n_offsprings, pair_params, mutate_params, init_params, param_adjustment=None):
        self.pair_params = pair_params
        self.mutate_params = mutate_params
        self.pool = Population(pool_size, fitness, individual_class, init_params)
        self.n_offsprings = n_offsprings
        self.epoch = 0
        self.param_adjustment = param_adjustment

    def step(self):
        mothers, fathers = self.pool.get_parents(self.n_offsprings)
        offsprings = []

        for mother, father in zip(mothers, fathers):
            offspring = mother.pair(father, self.pair_params)
            offspring.mutate(self.mutate_params)
            offsprings.append(offspring)

        self.pool.replace(offsprings)
        self.epoch += 1
        if self.param_adjustment:
            self.mutate_params = self.param_adjustment(self.mutate_params, self.epoch)
        

class TSP(Individual):
    def pair(self, other, pair_params):
        self_head = self.value[:int(len(self.value) * pair_params['alpha'])].copy()
        self_tail = self.value[int(len(self.value) * pair_params['alpha']):].copy()
        other_tail = other.value[int(len(other.value) * pair_params['alpha']):].copy()

        mapping = {other_tail[i]: self_tail[i] for i in range(len(self_tail))}

        for i in range(len(self_head)):
            while self_head[i] in other_tail:
                self_head[i] = mapping[self_head[i]]

        return TSP(np.hstack([self_head, other_tail]))

    def mutate(self, mutate_params):
        for _ in range(mutate_params['rate']):
            i, j = np.random.choice(range(len(self.value)), 2, replace=False)
            self.value[i], self.value[j] = self.value[j], self.value[i]

    def _random_init(self, init_params):
        return np.random.choice(range(init_params['n_cities']), init_params['n_cities'], replace=False)
    

class MP(Individual):
    def mutate(self, mutate_params):
        self.value += np.array([np.random.normal(loc=0, scale=mutate_params['std'], size=(3,3))
                                for _ in range(mutate_params['dim'])])
    
    def pair(self, other, pair_params):
        return MP(pair_params['alpha'] * self.value + (1-pair_params['alpha']) * other.value)
    
    def _random_init(self, init_params):
        return np.array([np.array([[1,0,0],[0,1,0],[0,0,1]]) + np.random.normal(loc=0, scale=init_params['std'], size=(3,3))
                         for _ in range(init_params['dim'])])
        

class MP_lin(Individual):
    def mutate(self, mutate_params):
        self.value += np.array([np.random.normal(scale=mutate_params['std'])
                                for _ in range(mutate_params['dim'])])
    
    def pair(self, other, pair_params):
        return MP_lin(pair_params['alpha'] * self.value + (1-pair_params['alpha']) * other.value)
    
    def _random_init(self, init_params):
        return np.array([np.random.normal(scale=init_params['std']) for _ in range(init_params['dim'])])


class Hemisphere(Individual):
    """
    Individual for finding the axis of the hemisphere for Alice and Bob's output and their biases.
    The values are the (hemisphere axis vector, bias).
    """
    def mutate(self, mutate_params):
        self.value = (self.value[0] + np.random.normal(scale=mutate_params['std'], size=(3,)),
                      self.value[1] + np.random.normal(scale=mutate_params['std']))
        self.value = (self.value[0]/np.linalg.norm(self.value[0]), self.value[1])
    
    def pair(self, other, pair_params):
        vec = pair_params['alpha'] * self.value[0] + (1-pair_params['alpha']) * other.value[0]
        return Hemisphere(
            (vec/np.linalg.norm(vec),
             pair_params['alpha'] * self.value[1] + (1-pair_params['alpha']) * other.value[1]))
    
    def _random_init(self, init_params):
        value = (np.random.normal(scale=init_params['std'], size=(3,)),
                    np.random.normal(scale=init_params['std']))
        return (value[0]/np.linalg.norm(value[0]), value[1])
    

class Weighting(Individual):
    """
    Individual for finding the weight for the analytical protocol of Alice and Bob's output.
    The values are the (weight of lambda_1, drag of z).
    """
    def mutate(self, mutate_params):
        self.value = (self.value[0] + np.random.normal(scale=mutate_params['std']),
                      self.value[1] + np.random.normal(scale=mutate_params['std']))
    
    def pair(self, other, pair_params):
        weight = pair_params['alpha'] * self.value[0] + (1-pair_params['alpha']) * other.value[0]
        drag = pair_params['alpha'] * self.value[1] + (1-pair_params['alpha']) * other.value[1]
        return Weighting((weight, drag))
    
    def _random_init(self, init_params):
        value = (np.random.normal(loc=init_params['loc_weight'], scale=init_params['std']),
                 np.random.normal(loc=init_params['loc_drag'], scale=init_params['std']))
        return (value[0]/np.linalg.norm(value[0]), value[1])
    

class Bias(Individual):
    """
    Individual for finding the bias for the analytical protocol of Alice and Bob's output.
    The value is the bias.
    """
    def mutate(self, mutate_params):
        self.value = self.value + np.random.normal(scale=mutate_params['std'])
    
    def pair(self, other, pair_params):
        bias = pair_params['alpha'] * self.value + (1-pair_params['alpha']) * other.value
        return Bias(bias)
    
    def _random_init(self, init_params):
        value = np.random.normal(loc=init_params['loc'], scale=init_params['std'])
        return value


class Bias_coefficients(Individual):
    """
    Individual for finding the regression for an expression of the bias as a function of the two LHVs.
    The value is the w,x,y where bias = w + x * lhv_1[2] + y * lhv_2[2].
    """
    def mutate(self, mutate_params):
        self.value = self.value + np.random.normal(scale=mutate_params['std'], size=3)
    
    def pair(self, other, pair_params):
        bias = pair_params['alpha'] * self.value + (1-pair_params['alpha']) * other.value
        return Bias_coefficients(bias)
    
    def _random_init(self, init_params):
        value = (np.random.normal(loc=init_params['loc'], scale=init_params['std'], size=3))
        return value
    

class Comm_coefficients(Individual):
    """
    Individual for finding the coefficients for the analytical protocol of the bit of communication.
    The value is u,v where b_c = u + v(lhv_2[2])(1-lhv_1[2]).
    """
    def mutate(self, mutate_params):
        self.value = self.value + np.random.normal(scale=mutate_params['std'], size=2)
    
    def pair(self, other, pair_params):
        bias = pair_params['alpha'] * self.value + (1-pair_params['alpha']) * other.value
        return Comm_coefficients(bias)
    
    def _random_init(self, init_params):
        value = (np.random.normal(loc=init_params['loc'], scale=init_params['std'], size=2))
        return value
