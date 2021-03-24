import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import config
from distribution_generator import CHSH_measurements_extended
from training import werner_run


""" # Will run the mixed run and werner test
config.training_size = 16
config.shuffle_epochs = 5
config.cutoff = 1e-3
mixed_run(n=256, start=0, end=1, step=11, a=None, b=None, generate_new=True)

config.training_size = 4
config.shuffle_epochs = 100
config.cutoff = 1e-4
a, b = CHSH_measurements()
werner_run(n=4, start=0, end=1, step=15, a=a,
           b=b, generate_new=True, comm=True)
werner_run(n=4, start=0.6, end=0.8, step=15, a=a,
           b=b, generate_new=True, comm=True)
"""

config.training_size = 8
config.shuffle_epochs = 100
config.cutoff = 1e-4
a, b = CHSH_measurements_extended()
werner_run(n=8, start=0, end=1, step=11, a=a,
           b=b, generate_new=True, comm=False)
werner_run(n=8, start=0, end=1, step=11, a=a,
           b=b, generate_new=True, comm=True)
