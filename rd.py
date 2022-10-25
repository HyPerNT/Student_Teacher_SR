"""Rough Draft for Interpretable Student-Teacher NNs on SR

Authors: Brenton Candelaria and Sophia Novo-Gradac

Testing ground for code, has some solid work done in advance to make more tedious work easier in the future

"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from eval import Eval, EvalPt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {‘0’, ‘1’, ‘2’}, stops tf from complaining about things it shouldn't be worried about
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

FIG_DIR = "figures"

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
        self.shape = (dim, )
        if gen_data:
            self.gen_data(sample_size=sample_size, scale=scale, test_size=test_size, noisy=noisy, noise_factor=noise_factor)

    def gen_data(self, sample_size=DEFAULT_SAMPLE_SIZE, scale=DEFAULT_SCALE, test_size=DEFAULT_TEST_SIZE, noisy=False, noise_factor=DEFAULT_NOISE, sorted=False):
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
                sorted : Whether to sort the data, useful for plots. Should also only be used on fns with a single variable
                """

        r = np.random.rand(int(sample_size*(1-test_size)), self.dim) * scale # Generate input train array according to params
        self.x_train = r
        if sorted:
            self.x_train = np.arange(0, scale, scale/sample_size)
            self.x_train = np.reshape(self.x_train, (len(self.x_train), 1))
        self.y_train = Eval(self.fn, self.x_train)[:,-1] # Init x_train and y_train

        r = np.random.rand(int(sample_size*test_size), self.dim) * scale # Generate seperate test array according to params
        self.x_test = r
        self.y_test = Eval(self.fn, self.x_test)[:,-1]  # Also init x_test and y_test
        
        # The below just adds noise as specified, centered at 0
        if noisy:
            noise = np.random.rand(int(sample_size * (1 - test_size)), self.dim)
            noise -= 0.5
            self.x_train = self.x_train + noise_factor * noise
            for i in self.y_train:
                i = i + noise_factor * (np.random.rand((0)) - 0.5)

            noise = np.random.rand(int(sample_size * test_size), self.dim)
            noise -= 0.5
            self.x_test = self.x_test + noise_factor * noise
            for i in self.y_test:
                i = i + noise_factor * (np.random.rand((0)) - 0.5)


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

def iterateNN(fn, layer_range=10, node_range=512, by=2, method="*", rate=0.01, name='nn', summary=False):
    """Generates many NNs as candidates for training, useful to find "an ideal" architecture before we craft a bad one by hand

        Parameters
        ----------
            fn : The expression to use for training (Constant across all NNs)
            layer_range : Number of layers to vary over, incrementally
            node_range : Number of nodes to vary over, incrementally or multiplicatively
            by : Factor to use when adding more nodes, either additive or multiplicative
            method : Whether to add more nodes "+", or to multiply # of nodes "*"
            rate : Dropout rate
            name : Name for each NN, used as a template since we append details about architecture to the name
            summary : Whether to print a brief summary of each model, might be useful

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
            model = getNN(i, j, fn.shape, rate, name) # Build a NN of that architectyre
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error') # Compile it
            train = model.fit(fn.x_train, fn.y_train, epochs=EPOCHS) # Train
            train = min(train.history['loss'])
            predict = model.evaluate(fn.x_test, fn.y_test) # Test
            if summary:
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
    # start = timeit.default_timer()

    # fn1 = mystery_function("0>>x{0}0>>x{1}^*/", 2, sample_size=100_000, scale=10, gen_data=True, noisy=True) # This is just 1/2 mv^2, basically
    sin = mystery_function("x{0}S", 1, scale=4, gen_data=True, noisy=True)

    # records, best_nn, all_models = iterateNN(sin, layer_range=10, name='sin_nn', node_range=512) # Store results from iteration

    # Build and train sin, playing with non-linear function learning
    nn = getNN(3, 512, sin.shape, name="sin")
    nn.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
    nn.fit(sin.x_train, sin.y_train, epochs=EPOCHS)
    nn.evaluate(sin.x_test, sin.y_test)


    # stop = timeit.default_timer()

    # Print everything out
    # TODO: Process all of the f*cking data to hopefully get some insight
    # print(f"Best models: \n{('-'*32)}\n{records}\n\n")
    # print(f"All models: \n{('-'*32)}\n{all_models}\n\n")
    
    
    # minutes, sec = divmod(stop-start, 60.0)
    # print(f"Took {int(minutes)} minutes, {sec:1.4f} seconds")
    
    
    # Seems like the "big" set of 10 layers, 2-512 nodes by powers of 2 for KE takes about 10 mins on my laptop - B
    
    # A note on why I'm brute-forcing NNs: Ramyaa suggested it would be a good idea to "get a feel" for the difficulty/
    # architecture of learning different polynomials. I'll craft more mystery fn's for us to work with and let sit + spin
    # Overnight eventually. I'll also write out the data so that we don't need to re-train every single time

    # Rn, this is just looking for the nearest matches on "best" performance
    # for i in records:
    #     if i[1] <= 5 * records[-1][1]:
    #         print(i)

    # for i in all_models:
    #     if i[1] <= 5 * records[-1][1]:
    #         print(i)

    # Across several tests, [2, 128] seems to be the best configuration for KE
    # Additionally, [3, 512] seems best for sin(x), though it doesn't seem to generalize outside of the training domain

    # Let's visualize the result we get for the best NN

    sin.gen_data(test_size=0, sample_size=500, scale=10, sorted=True)
    x = sin.x_train
    y1 = sin.y_train
    y2 = nn.predict(sin.x_train)
    x = np.reshape(x, (-1))
    y1 = np.reshape(y1, (-1))
    y2 = np.reshape(y2, (-1))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot(x, y1, label = "Ground Truth", color = "blue")
    plt.plot(x, y2, label = "Predicted", color = "red")
    err = y2-y1
    plt.plot(x, err, label = "Error", linestyle = "--", color = "green")


    plt.legend()
    plt.savefig(f'./{FIG_DIR}/{nn.name}.png')

if __name__ == "__main__":
    main()