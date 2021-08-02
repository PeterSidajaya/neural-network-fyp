import pandas as pd
import numpy as np
import random
import config
from distribution_generator import random_unit_vector, random_semicircle_vector

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
    return process_dataset(df, limit=limit)


def process_dataset(dataframe, limit=None):
    dataframe = dataframe.to_numpy()
    # input is the first 6 columns
    input_array = dataframe[::4, :6]
    # reshape the probability
    output_array = dataframe[:, 8].reshape(-1, 4)
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
    if config.LHV_type == "gauss":
        config.number_of_LHV = 1
        LHV_list = np.array([random.gauss(0.5, 0.28867) for i in range(
            LHV_per_setting * input_size * config.number_of_LHV)]).reshape(LHV_per_setting * input_size, -1)
    elif config.LHV_type == "uniform":
        config.number_of_LHV = 1
        LHV_list = np.array([random.uniform(0.0, 1.0) for i in range(
            LHV_per_setting * input_size * config.number_of_LHV)]).reshape(LHV_per_setting * input_size, -1)
    elif config.LHV_type == "vector":
        config.number_of_LHV = 3
        LHV_list = np.array([random_unit_vector(3) for i in range(
            LHV_per_setting * input_size)])
    elif config.LHV_type == "vector pair":
        config.number_of_LHV = 6
        LHV_list = np.array([np.concatenate((random_unit_vector(3), random_unit_vector(3))) for i in range(
            LHV_per_setting * input_size)])
    elif config.LHV_type == "symmetry vector":
        config.number_of_LHV = 2
        LHV_list = np.array([random_semicircle_vector() for i in range(
            LHV_per_setting * input_size)])
    else:
        raise ValueError('LHV type is not recognized.')
    return np.concatenate((input_array, LHV_list), axis=1)
