"""Rough Draft for Interpretable Student-Teacher NNs on SR

Authors: Brenton Candelaria and Sophia Novo-Gradac

Testing ground for code, has some solid work done in advance to make more tedious work easier in the future

"""

import numpy as np
import matplotlib as plt
from eval import Eval, EvalPt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {‘0’, ‘1’, ‘2’}
import tensorflow as tf
import re # Can't believe I'm actually using this
import timeit # Just so I know how much time I've wasted

# Some constants that can be tweaked
BIG_LAYER = 512
SMALL_LAYER = 64
BABY_LAYER = 16
EPOCHS = 10
DEFAULT_NOISE = 0.000001
DEFAULT_SAMPLE_SIZE = 10_000
DEFAULT_SCALE = 1
DEFAULT_TEST_SIZE = 0.2

# Maybe we can identify certain operations?
# Identity is always identity matrix
# We can preserve arguments by passing forward args as identity matrix forward to next layer, same with operations we've gathered
# Lin. combinations look like [ ... a b c.... d] for something like ax_n + bx_n+1 + cx_n+2 + dx_m
# ... I can't figure out how to do multiplication of two variables in an array...


class mystery_function():
    """A class to model a given fn, also handles creation/storage of data to train/test on"""
    def __init__(self, fn, dim, gen_data=False, sample_size=DEFAULT_SAMPLE_SIZE, scale=DEFAULT_SCALE, test_size=DEFAULT_TEST_SIZE, noisy=False, noise_factor=DEFAULT_NOISE):
        """Init function, generates a function for fn in dim variables, includes optional params for tuning the data generated
        
            Parameters
            ----------
                fn : The function, as a string, to be modeled
                dim : The number of variables in the function, typically just the length of each vector that would be in the input space
                gen_data : Whether to generate data on init, False by default
                sample_size : The number of samples to generate in total, configurable at head of this file
                scale : Scale/range to vary the Gaussian distribution of samples by, [0, 1) by default
                test_size : Portion of sample_size to be used for test batch, should be a float in [0, 1]
                noise : Whether to introduce Gaussian noise to the generated dataset, false by default
                noise_factor : Scale of Gaussian noise added when it is introduced
                """
        self.fn = fn
        self.dim = dim
        if gen_data:
            self.gen_data(fn, sample_size=sample_size, scale=scale, test_size=test_size, noisy=noisy, noise_factor=noise_factor)

    def gen_data(self, fn, sample_size=DEFAULT_SAMPLE_SIZE, scale=DEFAULT_SCALE, test_size=DEFAULT_TEST_SIZE, noisy=False, noise_factor=DEFAULT_NOISE):
        """
        Method to generate data for our candidate fn, makes a random numpy array and defers to Eval() for answers

            Parameters
            ----------
                fn : The function, as a string, to be modeled
                dim : The number of variables in the function, typically just the length of each vector that would be in the input space
                gen_data : Whether to generate data on init, False by default
                sample_size : The number of samples to generate in total, configurable at head of this file
                scale : Scale/range to vary the Gaussian distribution of samples by, [0, 1) by default
                test_size : Portion of sample_size to be used for test batch, should be a float in [0, 1]
                noise : Whether to introduce Gaussian noise to the generated dataset, false by default
                noise_factor : Scale of Gaussian noise added when it is introduced
                """

        r = np.random.rand(int(sample_size*(1-test_size)), self.dim) * scale # Generate input train array according to params
        self.x_train = r
        self.y_train = Eval(fn, r)[:,-1] # Init x_train and y_train

        r = np.random.rand(int(sample_size*test_size), self.dim) * scale # Generate seperate test array according to params
        self.x_test = r
        self.y_test = Eval(fn, r)[:,-1]  # Also init x_test and y_test
        
        # The below just adds noise as specified, centered at 0
        if noisy:
            self.x_train = self.x_train + noise_factor * np.random.rand(int(sample_size * (1 - test_size)), self.dim) - 0.5      
            for i in self.y_train:
                i = i + noise_factor * np.random.rand((0)) - 0.5

            self.x_test = self.x_test + noise_factor * np.random.rand(int(sample_size * test_size), self.dim) - 0.5 
            for i in self.y_test:
                i = i + noise_factor * np.random.rand((0)) - 0.5


def getNN(layers, nodes, in_shape, rate=0.01, name='nn'):
    """Builds a simple FF-NN with the n layers and m nodes per layer, as specified.
        Each layer is really a densely connected layer followed by a dropout to prevent overfitting, but details

        Parameters
        ----------
            layers : Number of layers the network should have
            nodes : Number of nodes per layer
            in_shape : Input shape of the NN
            rate : Dropout rate, 0.01 by default
            name : Name for the network, useful for tracking many NNs
    """
    # Make template NN
    model = tf.keras.Sequential([tf.keras.Input(shape=in_shape)],name=name)

    # Iteratively add layers
    for _ in range(layers):
        model.add(tf.keras.layers.Dense(nodes, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=rate))

    # Add final layer for regression
    model.add(tf.keras.layers.Dense(1))
    return model

def iterateNN(in_shape, fn, layer_range=10, node_range=512, by=2, method="*", rate=0.01, name='nn'):
    """Generates many NNs as candidates for training, useful to find "an ideal" architecture before we craft a bad one by hand

        Parameters
        ----------
            in_shape : Shape of the input layer
            fn : The expression to use for training (Constant across all NNs)
            layer_range : Number of layers to vary over, incrementally
            node_range : Number of nodes to vary over, incrementally or multiplicatively
            by : Factor to use when adding more nodes, either additive or multiplicative
            method : Whether to add more nodes "+", or to multiply # of nodes "*"
            rate : Dropout rate
            name : Name for each NN, used as a template since we append details about architecture to the name

        Returns
        -------
            A tuple, of the following form:
                (x, y, z)
            
            x : A list of the best tracked architectures (might reveal patterns)
            y : The best model produced
            z : A full history of all model performance

            Each entry of x is a list like:
                [  [i, j]  ,   loss  ]

            Where:
                i : Number of layers
                j : Number of nodes / layer
                loss : Loss of the model

            Each entry of z is a list like:
                [   [i, j]   ,   loss   ,   params   ]

            Where:
                i : Number of layers
                j : Number of nodes / layer
                loss : Loss of the model
                params : Number of configurable params of that model

    """

    best_nn_loss = -1
    best_ij = [-1,-1]
    best_confs = []
    name_base = name
    best_model = None
    history = []
    # Init all vars for tracking to bunk data for now

    # Vary across layers first
    for i in range(layer_range):
        j = 1
        # Vary across # of nodes per layer
        while j <= node_range:
            name = name_base + f"_{i}-layers_{j}-nodes" # Generate name for this model, describe it so we can see something useful in terminal
            print(f'\n\nTraining {name}...\n{(len(name)+12)*"-"}')
            model = getNN(i, j, in_shape, rate, name) # Build a NN of that architectyre
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error') # Compile it
            train = model.fit(fn.x_train, fn.y_train, epochs=EPOCHS) # Train
            train = min(train.history['loss'])
            predict = model.evaluate(fn.x_test, fn.y_test) # Test

            model.summary() # Print details, more terminal output

            history.append([[i,j], train+predict, model.count_params()]) # Append this model to the full history

            if train + predict < best_nn_loss or best_nn_loss == -1: # This model is the new best
                best_nn_loss = train + predict
                best_ij = [i,j]
                best_confs.append([best_ij, best_nn_loss]) # Append to best models
                best_model = model
                print(f"{(len(best_ij)+16)*'-'}\nNew Best: {best_ij}\nScore: {best_nn_loss:1.4f}\n{(len(best_ij)+16)*'-'}") # Log to term

            # Increase # of nodes as specified
            if method=="*":
                j *= by
            elif method=="+":
                j += by

    # Print best results, and return
    print(f"Best Configuration: {best_ij[0]} layers and {best_ij[1]} nodes/layer, loss of {best_nn_loss:1.4f}")
    return best_confs, best_model, history

def main():
    start = timeit.default_timer()
    fn1 = mystery_function("0>>x{0}0>>x{1}^*/", 2, sample_size=10_000, scale=10, gen_data=True, noisy=True) # This is just 1/2 mv^2, basically
    # sin = mystery_function("x{0}S", 1, noisy=True)
    records, best_nn, all_models= iterateNN((2,), fn1, layer_range=10, name='KE_nn') # Store results from iteration
    stop = timeit.default_timer()

    # Print everything out
    # TODO: Process all of the f*cking data to hopefully get some insight
    print(f"Best models: \n{('-'*32)}\n{records}\n\n")
    print(f"All models: \n{('-'*32)}\n{all_models}\n\n")
    minutes, sec = divmod(stop-start, 60.0)
    print(f"Took {int(minutes)} minutes, {sec:1.4f} seconds")

    for i in records:
        if i[2] <= 5 * records[-1][2]:
            print(i)

    for i in all_models:
        if i[2] <= 10 * records[-1][2]:
            print(i)

if __name__ == "__main__":
    main()