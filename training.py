from distribution_generator import generate_mixed, generate_werner, CHSH_measurements
from neural_network_util import build_model, build_model_comm, customLoss_multiple, comm_customLoss_multiple
from preprocess import open_dataset, add_LHV
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import qutip as qt
import matplotlib.pyplot as plt
import tensorflow as tf
import config


def train_model(dataset, limit=None):
    """Train a no-communication model
    """
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
    model.compile(loss=comm_customLoss_multiple, optimizer=optimizer, metrics=[])

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
    model.compile(loss=comm_customLoss_multiple, optimizer=optimizer, metrics=[])
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
        x_train = add_LHV(x_train)                       	# Add the LHV
        # To match the size of the inputs
        y_train = np.repeat(y_train, LHV_size, axis=0)
        history = model.fit(x_train, y_train, batch_size=training_size*LHV_size,
                            epochs=config.shuffle_epochs, verbose=1, shuffle=False)
        loss_history += history.history['loss']
        if min(history.history['loss']) < config.cutoff:          	# Cutoff at 1e-4
            break
    score = model.evaluate(x=x_train, y=y_train,
                           batch_size=data.shape[0]*LHV_size)
    return (min(score, min(history.history['loss'])), loss_history)


def train(model, dataset, save=False, save_name=None, lr=None, loss=None):
    """Train a communication model
    """
    print("Starting training (with communication)...")
    x, y = open_dataset(dataset)
    data = np.concatenate((x, y), axis=1).astype('float32')
    LHV_size = config.LHV_size
    training_size = config.training_size
    number_of_measurements = data.shape[0]

    print("Generating model...")
    K.clear_session()

    if lr:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = config.optimizer
    if loss:
        model.compile(loss=loss, optimizer=optimizer, metrics=[])
    else:
        model.compile(loss=comm_customLoss_multiple, optimizer=optimizer, metrics=[])
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
        x_train = add_LHV(x_train)                       	# Add the LHV
        # To match the size of the inputs
        y_train = np.repeat(y_train, LHV_size, axis=0)
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

