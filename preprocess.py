import pandas as pd
import numpy as np
import random


def open_dataset(filename):
    """Opens the csv of state behaviour and returns the inputs and outputs for the Neural Network training

    Args:
        filename (string): The name of the file containing the state behaviour

    Returns:
        (Array, Array): The first array in the tuple is the array for the inputs.
            The second array in the tuple is for the outputs. The size of the input is (-1, 6).
            The size of the output is (-1, 4).
    """
    df = pd.read_csv('datasets\ '[:-1] + filename, index_col=0)
    df = df.to_numpy()
    input_array = df[::4, :6]
    output_array = df[:, 8].reshape(-1, 4)
    return (input_array, output_array)


def add_LHV(input_array, LHV_per_setting=4096, num_of_LHV=1):
    """Adds LHV for every input setting

    Args:
        input_array (ndarray): the x_train data without any LHV
        LHV_per_setting (int, optional): number of LHV to be inserted. Defaults to 4096.

    Returns:
        ndarray: the x_train data with LHV
    """
    input_size = input_array.shape[0]
    input_array = np.repeat(input_array, LHV_per_setting, axis=0)
    LHV_list = np.array([random.gauss(0.5, 0.28867) for i in range(
        LHV_per_setting * input_size * num_of_LHV)]).reshape(LHV_per_setting * input_size, -1)
    return np.concatenate((input_array, LHV_list), axis=1)


def evaluate(model, x_train, y_train, LHV_per_setting=4096, n=1):
    print(f"LHV per setting = {LHV_per_setting}")
    results = np.ndarray((0, 4))
    for i in range(n):
        x_predict = model.predict(
            x_train[LHV_per_setting*i:LHV_per_setting*(i+1)]).mean(axis=0)
        eval = np.reshape((x_predict - y_train[LHV_per_setting*i]), (1, 4))
        results = np.concatenate((results, eval), axis=0)
    return results
