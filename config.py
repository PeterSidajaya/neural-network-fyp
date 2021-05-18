import tensorflow as tf

"""
This file contains all the settings and hyperparameters to be used for the neural network.
"""


LHV_size = 6000		    # number of LHV for evaluation
training_size = 4		# number of measurement settings for one training step
number_of_LHV = 1		# number of LHV (one means a single number)
LHV_type = "uniform"    # type of LHV ("gauss", "uniform")

party_width = 50			        # width of NN
party_depth = 3	    		        # depth of NN
party_outputsize = 2		        # size of output
activation_func = 'relu'            # activation function for NN
# activation function for the final layer of the comm part
activation_func_comm = 'sigmoid'

optimizer = tf.keras.optimizers.Nadam()		# just use Nadam
epochs = 1200								# should be enough
shuffle_epochs = 100                        # number of epochs before shuffling
# cutoff training when loss dips below this value
cutoff = 1e-4
