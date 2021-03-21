import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from distribution_generator import generate_different_settings, generate_mixed, generate_werner
from neural_network_util import *
import qutip as qt
import matplotlib.pyplot as plt
from colour import Color


def plot_measurements(num):
    num = min((num_of_settings, num))
    b = qt.Bloch()
    b.vector_color = list(map(lambda x: x.rgb, list(
        Color("red").range_to(Color("purple"), num))))
    vector_alice_list = x_train[:total_LHV*num:total_LHV, 0:3]
    vector_bob_list = x_train[:total_LHV*num:total_LHV, 3:6]
    for i in range(num):
        vector_alice = vector_alice_list[i, :]
        vector_bob = vector_bob_list[i, :]
        b.add_vectors(vector_alice)
        b.add_vectors(vector_bob)
    b.show()
    plt.show()


def single_evaluation(model, x_train, batch_size, total_LHV, index=0):
    """ Evaluates the model and returns the resulting distribution as a numpy array. """
    x_train = x_train[index*total_LHV:(index+1)*total_LHV]
    test_pred = model.predict(x_train, batch_size=batch_size)
    result = K.eval(customLoss_distr(test_pred))
    return result


def single_run(dataset, epochs=10):
    print("Running on data...")
    x, y = open_dataset(dataset)
    data = np.concatenate((x, y), axis=1)
    batch_size = 6000
    batches_per_setting = 1
    total_LHV = int(batch_size * batches_per_setting)

    print("Generating model...")
    K.clear_session()
    model = build_model()

    optimizer = tf.keras.optimizers.Adadelta(
        lr=2, rho=0.95, epsilon=None, decay=0.001)
    model.compile(loss=customLoss, optimizer=optimizer, metrics=[])

    rng = np.random.default_rng()
    print("Fitting model...")
    # Fit model
    for epoch in range(epochs):
        print("Shuffling data")
        rng.shuffle(data)
        x_train = data[:, :6]
        y_train = data[:, 6:]
        x_train = add_LHV(x_train, LHV_per_setting=total_LHV)
        y_train = np.repeat(y_train, total_LHV, axis=0)
        model.fit(x_train, y_train, batch_size=batch_size,
                  epochs=1, verbose=1, shuffle=False)
    return model.evaluate(x_train, y_train, batch_size)


def single_run_multiple(dataset, epochs=10):
    print("Running on data... (multiple)")
    x, y = open_dataset(dataset)
    data = np.concatenate((x, y), axis=1)
    batch_size = 6000
    batches_per_setting = 1
    total_LHV = int(batch_size * batches_per_setting)

    print("Generating model...")
    K.clear_session()
    model = build_model()
        
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=customLoss_multiple, optimizer=optimizer, metrics=[])

    rng = np.random.default_rng()
    print("Fitting model...")
    # Fit model
    for epoch in range(epochs):
        # print("Shuffling data")
        # rng.shuffle(data)
        x_train = data[:, :6]
        y_train = data[:, 6:]
        x_train = add_LHV(x_train, LHV_per_setting=total_LHV)
        y_train = np.repeat(y_train, total_LHV, axis=0)
        history = model.fit(x_train, y_train, batch_size=4*batch_size,
                            epochs=1, verbose=1, shuffle=False)
        if history.history['loss'][-1] < 1e-3:
            break
    return min(history.history['loss'])


def single_run_multiple_comm(dataset, epochs=10):
    print("Running on data... (multiple)")
    x, y = open_dataset(dataset)
    data = np.concatenate((x, y), axis=1)
    batch_size = 6000
    batches_per_setting = 1
    total_LHV = int(batch_size * batches_per_setting)

    print("Generating model...")
    K.clear_session()
    model = build_model_comm()

    optimizer = tf.keras.optimizers.Nadam()
    model.compile(loss=customLoss_multiple, optimizer=optimizer, metrics=[])

    rng = np.random.default_rng()
    print("Fitting model...")
    # Fit model
    for epoch in range(epochs):
        print("Shuffling data")
        rng.shuffle(data)
        x_train = data[:, :6]
        y_train = data[:, 6:]
        x_train = add_LHV(x_train, LHV_per_setting=total_LHV)
        y_train = np.repeat(y_train, total_LHV, axis=0)
        history = model.fit(x_train, y_train, batch_size=4*batch_size,
                            epochs=1, verbose=1, shuffle=False)
        if history.history['loss'][-1] < 1e-3:
            break
    return min(history.history['loss'])

def werner_run(n=1024, step=10, a=None, b=None, generate_new=True, epochs=10):
    if generate_new:
        print('Generating datasets...')
        generate_werner(n=n, step=step, a=a, b=b)
        print('Finished generating, starting training...')
    w_array = []
    loss_array = []
    count = 0
    for w in np.linspace(0, 1, step):
        print('Step {} out of {}'.format(count+1, step))
        filename = 'dataset_werner_state_' + str(count) + '.csv'
        w_array.append(w)
        loss_array.append(single_run(filename, epochs=epochs))
        count += 1
    plt.plot(w_array, loss_array)
    plt.title('Werner state loss')
    plt.xlabel('Werner parameter')
    plt.ylabel('Relative entropy')
    plt.show()


def mixed_run(n=1024, step=10, a=None, b=None, generate_new=True, epochs=10):
    if generate_new:
        print('Generating datasets...')
        generate_mixed(n=n, step=step, a=a, b=b)
        print('Finished generating, starting training...')
    w_array = []
    loss_array = []
    count = 0
    for w in np.linspace(0, 1, step):
        print('Step {} out of {}'.format(count+1, step))
        filename = 'dataset_mixed_separable_state_' + str(count) + '.csv'
        w_array.append(w)
        loss_array.append(single_run(filename, epochs=epochs))
        count += 1
    plt.plot(w_array, loss_array)
    plt.title('Mixed separable state')
    plt.xlabel('Mixing parameter')
    plt.ylabel('Relative entropy')
    plt.show()


def CHSH_measurements():
    vec_1 = [0, 0, 1]
    vec_2 = [1, 0, 0]
    vec_3 = [1/np.sqrt(2), 0, 1/np.sqrt(2)]
    vec_4 = [-1/np.sqrt(2), 0, 1/np.sqrt(2)]

    a = [vec_1, vec_1, vec_2, vec_2]
    b = [vec_3, vec_4, vec_3, vec_4]
    return (a, b)


def werner_run_small(n=4, step=10, a=None, b=None, generate_new=True, epochs=600):
    if generate_new:
        print('Generating datasets...')
        generate_werner(n=n, step=step, a=a, b=b)
        print('Finished generating, starting training...')
    min_loss = [1e5,] * step
    for i in range(1):
        w_array = []
        loss_array = []
        count = 0
        for w in np.linspace(0, 1, step):
            print('Step {} out of {}'.format(count+1, step))
            filename = 'dataset_werner_state_' + str(count) + '.csv'
            w_array.append(w)
            loss_array.append(single_run_multiple(filename, epochs=epochs))
            count += 1
        plt.plot(w_array, loss_array, 'x')
        for i in range(step):
            if min_loss[i] > loss_array[i]:
                min_loss[i] = loss_array[i]
    plt.plot(w_array, min_loss, '-o')
    plt.title('Werner state loss')
    plt.xlabel('Werner parameter')
    plt.ylabel('Relative entropy')
    plt.show()
    df = pd.DataFrame({'werner_parameter': w_array, 'loss': min_loss})
    df.to_csv('werner_state.csv')


def communication_test(start=1, end=100, step=5, epochs=500):
    print('Generating datasets...')
    generate_different_settings(start=start, end=end, step=step)
    print('Finished generating, starting training...')
    for i in range(3):
        n_array = []
        loss_array = []
        loss_array_comm = []
        count = 0
        for n in np.linspace(start, end, step):
            print('Step {} out of {}'.format(count+1, step))
            filename = 'dataset_non_maximally_entangled_pi8_state_' + \
                str(count) + '.csv'
            n_array.append(4*int(n))
            print('Without communication')
            loss_array.append(single_run_multiple(filename, epochs=1+int(epochs//n)))
            print('With communication')
            loss_array_comm.append(single_run_multiple_comm(filename, epochs=1+int(epochs//n)))
            count += 1
        plt.plot(n_array, loss_array, '--o')
        plt.plot(n_array, loss_array_comm, '--x')
    plt.title('Communication advantage comparison')
    plt.xlabel('Number of settings')
    plt.ylabel('Relative entropy')
    plt.show()


a, b = CHSH_measurements()
werner_run_small(n=4, step=10, a=a, b=b)
