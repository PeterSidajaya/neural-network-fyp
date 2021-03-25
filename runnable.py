import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import config
from distribution_generator import CHSH_measurements_extended
from training import werner_run
from grapher import plot_dataset

plot_dataset('werner_result_without_comm_CHSH_extended.csv', xlabel='Werner parameter',
             ylabel='Relative entropy', title='Werner state loss')
