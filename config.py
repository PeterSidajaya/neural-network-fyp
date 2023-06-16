import tensorflow as tf

"""
This file contains all the settings and hyperparameters to be used for the neural network.
"""


LHV_size = 1000             # number of LHV for evaluation
training_size = 400         # number of measurement settings for one training step
number_of_LHV = 6           # number of LHV (one means a single number)
LHV_type = "vector pair"    # type of LHV ("gauss", "uniform")

party_width = 50                    # width of NN
party_depth = 3                     # depth of NN
party_outputsize = 2                # size of output
activation_func = 'relu'            # activation function for NN
activation_func_comm = 'sigmoid'    # activation function for the final layer of the comm part

optimizer = tf.keras.optimizers.Adam()      # just use Adam
epochs = 100                                # should be enough
shuffle_epochs = 1                          # number of epochs before shuffling, deprecated, use generator instead
cutoff = 1e-4                               # cutoff training when loss dips below this value
