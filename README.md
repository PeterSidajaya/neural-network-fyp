# neural-network-fyp

Link to the paper: https://arxiv.org/abs/2305.19935

Neural network modified from: https://github.com/tkrivachy/neural-network-for-nonlocality-in-networks

Evolutionary algorithms taken from: https://towardsdatascience.com/an-extensible-evolutionary-algorithm-example-in-python-7372c56a557b

## Structure of the repo

The files are all in the `file` directory. The important files are
* `config.py`					: Contains constants for creating the neural network.
* `distribution_generator.py`	: Contains functions that calculate the quantum probabilities.
* `neural_network_util.py`	: The main file used to build the neural network.
* `training.py`				: Contains functions that help train the neural network.
* `model_testing.py`			: Files with functions that are used to analyse the model, plotting, etc.
* `evo.py`					: The main file for handling the evolutionary algorithm
* `runnable_training.py`		: The file called in the command line for the training phase.
* `runnable_testing.py`		: The file called in the command line for the testing phase. Now it's just used to plot.
* `runnable_evo.py`			: The file called in the command line to find the parameters for the semianalytical protocols using evolutionary algorithms.

The other files are either supporting files, or deprecated.

Additional directories:
* `qubits-new`	: This folder contains the models. Their errors are the blue circles in Fig. 2
* `qubits-simple`	: This folder contains the models which have simplified (single layer) communication network. Their errors are the red crosses in Fig. 2

For now, only the pi-4 and 5pi-32 are available. I will upload more in the future.

Documentation is an ongoing process. I will try to update the repo in the future.
