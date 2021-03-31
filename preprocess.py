import pandas as pd
import numpy as np
import random
import config

"""This file contains all the functions needed to preprocess the datasets.
"""


def open_dataset(filename, limit=None):
    """Opens the csv of state behaviour and returns the inputs and outputs for the Neural Network training

    Args:
        filename (string): The name of the file containing the state behaviour

    Returns:
        (Array, Array): The first array in the tuple is the array for the inputs.
            The second array in the tuple is for the outputs. The size of the input is (-1, 6).
            The size of the output is (-1, 4).
    """
    df = pd.read_csv(filename, index_col=0)
    df = df.to_numpy()
    input_array = df[::4, :6]                   # input is the first 6 columns
    output_array = df[:, 8].reshape(-1, 4)      # reshape the probability
    if limit:
        input_array = input_array[:limit]
        output_array = output_array[:limit]
    return (input_array, output_array)


def add_LHV(input_array):
    """Adds LHV for every input setting

    Args:
        input_array (ndarray): the x_train data without any LHV

    Returns:
        ndarray: the x_train data with LHV
    """
    LHV_per_setting = config.LHV_size
    input_size = input_array.shape[0]
    input_array = np.repeat(input_array, LHV_per_setting, axis=0)
    LHV_list = np.array([random.gauss(0.5, 0.28867) for i in range(
        LHV_per_setting * input_size * config.number_of_LHV)]).reshape(LHV_per_setting * input_size, -1)
    # LHV_list = np.array([random.uniform(0, 1) for i in range(
    #     LHV_per_setting * input_size * config.number_of_LHV)]).reshape(LHV_per_setting * input_size, -1)
    return np.concatenate((input_array, LHV_list), axis=1)
