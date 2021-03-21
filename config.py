import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

LHV_size = 6000		# number of LHV for evaluation
training_size = 4		# number of measurement settings for one training step
number_of_LHV = 1		# number of LHV (one means a single number)

party_width = 100			# width of NN
party_depth = 100			# depth of NN
party_outputsize = 2		# size of output
activation_func = 'relu'	# activation function for NN

optimizer = tf.keras.optimizer.Adam()		# just use Adam
epochs = 600								# should be enough