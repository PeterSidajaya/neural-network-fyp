from distribution_generator import generate_mixed, generate_werner, CHSH_measurements, random_unit_vector, random_unit_vectors
from neural_network_util import build_model, build_model_comm, customLoss_multiple, comm_customLoss_multiple, CGLMP_local, CGLMP_nonlocal
from preprocess import open_dataset, add_LHV
import distribution_generator
import distribution_generator_qutrit_spin
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import qutip as qt
import matplotlib.pyplot as plt
import tensorflow as tf
import config


def create_generator(state, dim=2):
    num_of_measurement = config.training_size
    LHV_size = config.LHV_size
    batch_size = num_of_measurement * LHV_size
    while True:
        x = np.ndarray((0, 6 + config.number_of_LHV))
        y = np.ndarray((0, dim**2))
        lhvs_1 = random_unit_vectors(LHV_size, 3).astype('float32')
        lhvs_2 = random_unit_vectors(LHV_size, 3).astype('float32')
        for i in range(num_of_measurement):
            vec_a, vec_b = random_unit_vector(3), random_unit_vector(3)

            if dim == 2:
                prob = distribution_generator.probability_list(state, vec_a, vec_b)
            if dim == 3:
                prob = distribution_generator_qutrit_spin.probability_list(state, vec_a, vec_b)
            probs = np.repeat(np.array([prob, ]), LHV_size, axis=0)
            y = np.concatenate((y, probs), axis=0).astype('float32')

            inputs = np.concatenate((np.repeat(
                [vec_a + vec_b, ], LHV_size, axis=0), lhvs_1, lhvs_2), axis=1)
            x = np.concatenate((x, inputs), axis=0).astype('float32')
        yield (x, y)


def create_generator_limited(state, alice_set, bob_set, dim=2):
    num_of_measurement = len(alice_set) * len(bob_set)
    config.training_size = num_of_measurement
    LHV_size = config.LHV_size
    batch_size = num_of_measurement * LHV_size
    rng = np.random.default_rng()
    
    while True:
        x = np.ndarray((0, 6 + config.number_of_LHV))
        y = np.ndarray((0, dim**2))
        lhvs_1 = random_unit_vectors(LHV_size, 3).astype('float32')
        lhvs_2 = random_unit_vectors(LHV_size, 3).astype('float32')
        for i in range(len(alice_set)):
            for j in range(len(bob_set)):
                vec_a, vec_b = np.array(alice_set[i]).astype('float32'), np.array(bob_set[j]).astype('float32')
                vec_a = vec_a / np.linalg.norm(vec_a)
                vec_b = vec_b / np.linalg.norm(vec_b)
                
                if dim == 2:
                    prob = distribution_generator.probability_list(state, vec_a, vec_b)
                if dim == 3:
                    prob = distribution_generator_qutrit_spin.probability_list(state, vec_a, vec_b)
                probs = np.repeat(np.array([prob, ]), LHV_size, axis=0)
                y = np.concatenate((y, probs), axis=0).astype('float32')

                inputs = np.concatenate((np.repeat(
                    np.array([np.array([vec_a, vec_b]).flatten(),]), LHV_size, axis=0), lhvs_1, lhvs_2), axis=1)
                x = np.concatenate((x, inputs), axis=0).astype('float32')
        yield (x, y)


def train(model, dataset, save=False, save_name=None, lr=None, loss=None, dim=2):
    """Train a communication model
    """
    print("Starting training (with communication)...")
    x, y = open_dataset(dataset, dim=dim)
    data = np.concatenate((x, y), axis=1).astype('float32')
    LHV_size = config.LHV_size
    training_size = config.training_size
    number_of_measurements = data.shape[0]

    K.clear_session()

    if lr:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = config.optimizer
    if loss:
        model.compile(loss=loss, optimizer=optimizer, metrics=[])
    else:
        model.compile(loss=comm_customLoss_multiple,
                      optimizer=optimizer, metrics=[])
    loss_history = []

    rng = np.random.default_rng()
    print("Fitting model...")
    # Fit model
    for epoch in range(config.epochs // config.shuffle_epochs):
        print("Shuffling data")
        print("Shuffle ", epoch + 1, " out of ",
              config.epochs // config.shuffle_epochs)
        rng.shuffle(data)
        x_train = data[:, :6]
        y_train = data[:, 6:]

        print("Adding LHV...")
        x_train = add_LHV(x_train)                       	# Add the LHV
        # To match the size of the inputs
        y_train = np.repeat(y_train, LHV_size, axis=0)

        print("Preparation finished, starting training...")
        history = model.fit(x_train, y_train, batch_size=training_size*LHV_size,
                            epochs=config.shuffle_epochs, verbose=1, shuffle=False)
        loss_history += history.history['loss']
        if history.history['loss'][-1] < config.cutoff:          	# Cutoff at 1e-4
            break

    # score = model.evaluate(x=x_train, y=y_train,
    #                        batch_size=data.shape[0]*LHV_size)
    if save:
        print("Saving model...")
        model.save(save_name)
    # return (min(score, min(history.history['loss'])), loss_history)
    return (min(history.history['loss']), loss_history)


def train_generator(model, generator, save=False, save_name=None, lr=None, loss=None, steps=50):
    """Train a communication model"""
    print("Starting training...")
    LHV_size = config.LHV_size
    training_size = config.training_size

    K.clear_session()

    if lr:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = config.optimizer
    if loss:
        model.compile(loss=loss, optimizer=optimizer, metrics=[])
    else:
        model.compile(loss=comm_customLoss_multiple,
                      optimizer=optimizer, metrics=[])

    # Fit model
    print("Fitting model with generator")
    history = model.fit(x=generator, batch_size=training_size*LHV_size,
                        epochs=config.epochs, steps_per_epoch=steps)
    loss_history = history.history['loss']

    # Save model
    if save:
        print("Saving model...")
        model.save(save_name)
    return (min(history.history['loss']), loss_history)


"""Warning, deprecated functions follow, thread on your own risk!"""


def train_model(dataset, limit=None):
    """Train a model"""
    print("Starting training...")
    x, y = open_dataset(dataset, limit=limit)
    data = np.concatenate((x, y), axis=1)
    LHV_size = config.LHV_size
    training_size = config.training_size
    number_of_measurements = data.shape[0]

    print("Generating model...")
    K.clear_session()
    model = build_model()

    optimizer = config.optimizer
    model.compile(loss=customLoss_multiple, optimizer=optimizer, metrics=[])

    rng = np.random.default_rng()
    print("Fitting model...")
    # Fit model
    for epoch in range(config.epochs // config.shuffle_epochs):
        print("Shuffling data")
        print("Shuffle ", epoch, " out of ",
              config.epochs // config.shuffle_epochs)
        rng.shuffle(data)
        x_train = data[:, :6]
        y_train = data[:, 6:]
        x_train = add_LHV(x_train)                       	# Add the LHV
        # To match the size of the inputs
        y_train = np.repeat(y_train, LHV_size, axis=0)
        history = model.fit(x_train, y_train, batch_size=training_size*LHV_size,
                            epochs=config.shuffle_epochs, verbose=1, shuffle=False)
        if min(history.history['loss']) < config.cutoff:          	# Cutoff at 1e-4
            break
    score = model.evaluate(x=x_train, y=y_train,
                           batch_size=data.shape[0]*LHV_size)
    return min(score, min(history.history['loss']))


def train_model_comm(dataset, limit=None):
    """Train a communication model
    """
    print("Starting training (with communication)...")
    x, y = open_dataset(dataset, limit=limit)
    data = np.concatenate((x, y), axis=1)
    LHV_size = config.LHV_size
    training_size = config.training_size
    number_of_measurements = data.shape[0]

    print("Generating model...")
    K.clear_session()
    model = build_model_comm()

    optimizer = config.optimizer
    model.compile(loss=comm_customLoss_multiple,
                  optimizer=optimizer, metrics=[])

    rng = np.random.default_rng()
    print("Fitting model...")
    # Fit model
    for epoch in range(config.epochs // config.shuffle_epochs):
        print("Shuffling data")
        print("Shuffle ", epoch + 1, " out of ",
              config.epochs // config.shuffle_epochs)
        rng.shuffle(data)
        x_train = data[:, :6]
        y_train = data[:, 6:]
        x_train = add_LHV(x_train)                       	# Add the LHV
        # To match the size of the inputs
        y_train = np.repeat(y_train, LHV_size, axis=0)
        history = model.fit(x_train, y_train, batch_size=training_size*LHV_size,
                            epochs=config.shuffle_epochs, verbose=1, shuffle=False)
        if min(history.history['loss']) < config.cutoff:          	# Cutoff at 1e-4
            break
    score = model.evaluate(x=x_train, y=y_train,
                           batch_size=data.shape[0]*LHV_size)
    return min(score, min(history.history['loss']))



def mixed_run(n=4, start=0, end=1, step=10, a=CHSH_measurements()[0], b=CHSH_measurements()[1], generate_new=True):
    """Train the no-communication model for a set of mixed states.
    Keywrord arguments:
            n: number of measurements; the training_size will be set to this value also (default 4)
            start: starting point of the mixing parameter (default 0)
            end: ending point of the mixing parameter (default 1)
            step: number of states in the set (default 10)
            a: optional list of vectors to be used as the measurement for Alice (default CHSH)
            b: optional list of vectors to be used as the measurement for Bob (default CHSH)
            generate_new: set to False if you do not want to generate new data
    """
    if a or b:
        n = len(a)
    if generate_new:
        print('Generating mixed state datasets...')
        generate_mixed(n=n, start=start, end=end, step=step, a=a, b=b)
        print('Finished generating, starting training...')
    # config.training_size = n 		# to do batch optimization
    w_array = []
    loss_array = []
    count = 0
    for w in np.linspace(start, end, step):
        print('Step {} out of {}'.format(count+1, step))
        filename = 'datasets\\dataset_mixed_separable_state_' + \
            str(count) + '.csv'
        w_array.append(w)
        loss_array.append(train_model(filename))
        count += 1
    print('Enter filename (without .csv): ')
    savename = input()
    df = pd.DataFrame({'mixing_parameter': w_array, 'loss': loss_array})
    df.to_csv(savename + '.csv')


def werner_run(n=4, start=0, end=1, step=10, a=CHSH_measurements()[0], b=CHSH_measurements()[1], comm=False, generate_new=True):
    """Train the no-communication model for a set of werner states.
    Keywrord arguments:
            n: number of measurements; the training_size will be set to this value also (default 4)
            start: starting point of the werner parameter (default 0)
            end: ending point of the werner parameter (default 1)
            step: number of states in the set (default 10)
            a: optional list of vectors to be used as the measurement for Alice (default CHSH)
            b: optional list of vectors to be used as the measurement for Bob (default CHSH)
            comm: whether to allow communication or not (default False)
            generate_new: set to False if you do not want to generate new data
    """
    if a or b:
        n = len(a)
    if generate_new:
        print('Generating werner state datasets...')
        generate_werner(n=n, start=start, end=end, step=step, a=a, b=b)
        print('Finished generating, starting training...')
    config.training_size = n 		# to do batch optimization
    w_array = []
    loss_array = []
    count = 0
    for w in np.linspace(start, end, step):
        print('Step {} out of {}'.format(count+1, step))
        filename = 'datasets\\dataset_werner_state_' + str(count) + '.csv'
        w_array.append(w)
        if comm:
            loss_array.append(train_model_comm(filename))
        else:
            loss_array.append(train_model(filename))
        count += 1
    print('Enter filename (without .csv): ')
    savename = input()
    df = pd.DataFrame({'werner_parameter': w_array, 'loss': loss_array})
    df.to_csv(savename + '.csv')
    
    
def CGLMP_training(noise=0.0, comm=False):
    def projector_alice(a,x):
        if x == 0:
            alpha = 0
        elif x == 1:
            alpha = np.pi/3
        ket = sum([np.exp(1j*2*np.pi/3*a*k) * np.exp(1j*k*alpha) / np.sqrt(3) * qt.basis(3,k) for k in range(3)])
        return qt.ket2dm(ket.unit())


    def projector_bob(b,y):
        if y == 0:
            beta = -np.pi/6
        elif y == 1:
            beta = np.pi/6
        ket = sum([np.exp(1j*2*np.pi/3*b*k) * np.exp(1j*k*beta) / np.sqrt(3) * qt.basis(3,k) for k in range(3)])
        return qt.ket2dm(ket.unit())


    def CGLMP_probability(i,j,noise):
        gamma = (np.sqrt(11) - np.sqrt(3))/2
        ket = (qt.tensor(qt.basis(3,0), qt.basis(3,0)) \
            + qt.tensor(qt.basis(3,1), qt.basis(3,1)) \
            + gamma * qt.tensor(qt.basis(3,2), qt.basis(3,2))).unit()
        state = noise * 1/9 * qt.tensor(qt.identity(3), qt.identity(3)) + (1-noise) * qt.ket2dm(ket)
        res = np.ndarray(9)
        for a in range(3):
            for b in range(3):
                P_a, P_b = projector_alice(a,i), projector_bob(b,j)
                res[3*a+b] = np.real((qt.tensor(P_a, P_b) * state).tr())
        return res
        
    if not comm:
        model = CGLMP_local()
    else:
        model = CGLMP_nonlocal()
    x = np.ndarray((4,2))
    y_true = np.ndarray((4,9))
    for i in range(2):
        for j in range(2):
           x[2*i+j,:] = [i,j]
           y_true[2*i+j,:] = CGLMP_probability(i,j,noise=noise)
    config.training_size = 4
    config.LHV_size = 6000
    config.LHV_type = 'uniform'
    x = add_LHV(x)
    y_true = np.repeat(y_true, config.LHV_size, axis=0)
    
    x = np.tile(x, (100, 1))
    y_true = np.tile(y_true, (100,1))
    
    if not comm:
        model.compile(loss=customLoss_multiple,
            optimizer=config.optimizer, metrics=[])
    else:
        model.compile(loss=comm_customLoss_multiple,
            optimizer=config.optimizer, metrics=[])
    history = model.fit(x, y_true, batch_size=4*config.LHV_size,
        epochs=50, verbose=0, shuffle=False)
    return history.history['loss'][-1]

for n in range(11):
    print("noise = ", 0.1*n)
    print(CGLMP_training(noise=0.1*n, comm=True))
           