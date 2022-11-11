"""Rough Draft for Interpretable Student-Teacher NNs on SR

Authors: Brenton Candelaria and Sophia Novo-Gradac

Testing ground for code, has some solid work done in advance to make more tedious work easier in the future

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {‘0’, ‘1’, ‘2’}, stops tf from complaining about things it shouldn't be worried about

import numpy as np
import matplotlib.pyplot as plt
from eval import Eval
import tensorflow as tf
import math
import logging
from datetime import datetime
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
DEFAULT_CENTER = 0
GREEDY_LOSS = 10**-6

FIG_DIR = "figures"
UNIT_DIR = "unit_fns"
LOG_DIR = 'log'

class mystery_function():
    """A class to model a given fn, also handles creation/storage of data to train/test on"""
    def __init__(self, fn, dim, gen_data=False, sample_size=DEFAULT_SAMPLE_SIZE, scale=DEFAULT_SCALE, center=DEFAULT_CENTER, test_size=DEFAULT_TEST_SIZE, noisy=False, noise_factor=DEFAULT_NOISE):
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
            self.gen_data(sample_size=sample_size, scale=scale, test_size=test_size, center=center, noisy=noisy, noise_factor=noise_factor)

    def gen_data(self, sample_size=DEFAULT_SAMPLE_SIZE, scale=DEFAULT_SCALE, center=DEFAULT_CENTER, test_size=DEFAULT_TEST_SIZE, noisy=False, noise_factor=DEFAULT_NOISE, sorted=False):
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
        center_factor = center - scale/2
        r = r + center_factor
        self.x_train = r
        if sorted:
            self.x_train = np.arange(center-scale/2, center+scale/2, scale/sample_size)
            self.x_train = np.reshape(self.x_train, (len(self.x_train), 1))
        self.y_train = Eval(self.fn, self.x_train)[:,-1] # Init x_train and y_train

        r = np.random.rand(int(sample_size*test_size), self.dim) * scale # Generate seperate test array according to params
        r = r + center_factor
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
            logging.info(f'\n\nTraining {name}...\n{(len(name)+12)*"-"}')
            model = getNN(i, j, fn.shape, rate, name) # Build a NN of that architectyre
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error') # Compile it
            train = model.fit(fn.x_train, fn.y_train, epochs=EPOCHS) # Train
            loss_history = train.history['loss']
            numpy_loss_history = np.array(loss_history)
            np.savetxt(f"./{LOG_DIR}/training_logs/{name_base}/{name}.txt", numpy_loss_history, delimiter=",")
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
                logging.info(f"{(len(best_ij)+16)*'-'}\nNew Best: {best_ij}\nScore: {best_nn_loss:1.4f}\n{(len(best_ij)+16)*'-'}") # Log to term
            # Increase # of nodes as specified
            if method=="*":
                j *= by
            elif method=="+":
                j += by
            if i == 0:
                j += node_range
            if best_nn_loss < GREEDY_LOSS:
                logging.info("Got greedy, found sufficient loss")
                logging.info(f"Best Configuration: {best_ij[0]} layers and {best_ij[1]} nodes/layer, loss of {best_nn_loss:1.4f}")
                return best_confs, best_model, history

    # Print best results, and return
    logging.info(f"Best Configuration: {best_ij[0]} layers and {best_ij[1]} nodes/layer, loss of {best_nn_loss:1.4f}")
    return best_confs, best_model, history

def plotNN(fn, nn):
    x = fn.x_train
    y1 = fn.y_train
    y2 = nn.predict(fn.x_train)
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




def bf_unit_nns():
    """This is meant to iterate through every unit funciton we intend to use and find an ideal enough architecture for unit students"""

    binaries = {0: ['+', 'addition'], 1: ['-', 'subtraction'], 2: ['*', 'multiplication'], 3: ['/', 'division'], 4: ['^', 'exponentiation']}
    unaries = {0: ['~', 'negation'], 1: ['A', 'abs'], 2: ['>', 'successor'], 3: ['<', 'predecessor']}
    ext = {0: ['L', 'log'], 1: ['S', 'sin'], 2: ['C', 'cos'], 3: ['s', 'arcsin'], 4: ['c', 'arccos'], 5: ['t', 'arctan']}


    if not os.path.exists(f'./{UNIT_DIR}'):
        os.mkdir(f'./{UNIT_DIR}')

    curr_time =datetime.now().strftime('%b-%d-%Y_%H%M%S')
    if os.path.getsize(f'./{LOG_DIR}/latest_run.log') > 10:
        os.rename(f'./{LOG_DIR}/training_logs/', f'./{LOG_DIR}/training_logs_{curr_time}/')
        os.rename(f'./{LOG_DIR}/latest_run.log', f'./{LOG_DIR}/latest_run_{curr_time}.log')
        os.rename(f'./{LOG_DIR}/tensorflow.log', f'./{LOG_DIR}/tensorflow_{curr_time}.log')

    if not os.path.exists(f'./{LOG_DIR}/training_logs'):
        os.mkdir(f'./{LOG_DIR}/training_logs')

    # First, run over binary operators
    for i in range(5):

        # Skip NNs we've found (allows us to select which NNs to re-train by renaming them/deleting them)
        path = os.path.join(UNIT_DIR, binaries[i][1])
        if os.path.exists(path):
            continue

        # Make directory for training logs
        if not os.path.exists(f'./{LOG_DIR}/training_logs/{binaries[i][1]}'):
            os.mkdir(f'./{LOG_DIR}/training_logs/{binaries[i][1]}')

        # Build the function / data
        center = 0
        if i == 4:
            center = 1
        fn = mystery_function(f"x{{0}}x{{1}}{binaries[i][0]}", 2, gen_data=True, sample_size=100_000, center=center, scale=2)

        # Brute Force
        records, best_nn, all_models = iterateNN(fn, name=f"{binaries[i][1]}")

        # Log results
        logging.info(f'Satisfactory models for {binaries[i][1]}')
        for k in all_models:
            if k[1] <= 5 * records[-1][1]:
                logging.info(f'Config:{k[0]}\tScore:{k[1]}\tParams:{k[2]}')

        # Save best NN
        os.mkdir(path)
        best_nn.save(path)


    # Run the simple unary fns
    for i in range(4):

        # Skip NNs we've found (allows us to select which NNs to re-train by renaming them/deleting them)
        path = os.path.join(UNIT_DIR, unaries[i][1])
        if os.path.exists(path):
            continue

        # Make directory for training logs
        if not os.path.exists(f'./{LOG_DIR}/training_logs/{unaries[i][1]}'):
            os.mkdir(f'./{LOG_DIR}/training_logs/{unaries[i][1]}')

        # Build the function / data
        fn = mystery_function(f"x{{0}}{unaries[i][0]}", 1, gen_data=True, sample_size=100_000, scale=2)

        # Brute Force
        records, best_nn, all_models = iterateNN(fn, name=f"{unaries[i][1]}")

        # Log results
        logging.info(f'Satisfactory models for {unaries[i][1]}')
        for k in all_models:
            if k[1] <= 5 * records[-1][1]:
                logging.info(f'Config:{k[0]}\tScore:{k[1]}\tParams:{k[2]}')

        # Save best NN
        os.mkdir(path)
        best_nn.save(path)

    # Run the scientific unary fns
    for i in range(6):

        # These next few lines simply adjust the domains for some fns to avoid getting undefined/NaN results during training
        center = 0
        scale = 10_000
        if i > 0 and i < 3:
            scale = 2 * math.pi
            center = math.pi
        if i > 2 and i < 5:
            scale = 2
        if i == 0:
            center = 5_000

        # Skip NNs we've found (allows us to select which NNs to re-train by renaming them/deleting them)
        path = os.path.join(UNIT_DIR, ext[i][1])
        if os.path.exists(path):
            continue

        # Make directory for training logs
        if not os.path.exists(f'./{LOG_DIR}/training_logs/{ext[i][1]}'):
            os.mkdir(f'./{LOG_DIR}/training_logs/{ext[i][1]}')


        # Build the function / data
        fn = mystery_function(f"x{{0}}{ext[i][0]}", 1, gen_data=True, sample_size=100_000, center=center, scale = scale)

        # Brute Force
        records, best_nn, all_models = iterateNN(fn, name=f"{ext[i][1]}")

        # Log results
        logging.info(f'Satisfactory models for {ext[i][1]}')
        for k in all_models:
            if k[1] <= 5 * records[-1][1]:
                logging.info(f'Config:{k[0]}\tScore:{k[1]}\tParams:{k[2]}')

        # Save best NN
        os.mkdir(path)
        best_nn.save(path)

    # Store more than the most recent logs, might be useful
    curr_time =datetime.now().strftime('%b-%d-%Y_%H%M%S')
    os.rename(f'./{LOG_DIR}/training_logs/', f'./{LOG_DIR}/training_logs_{curr_time}/')
    os.popen(f'cp ./{LOG_DIR}/latest_run.log ./{LOG_DIR}/latest_run_{curr_time}.log')
    os.popen(f'cp ./{LOG_DIR}/tensorflow.log ./{LOG_DIR}/tensorflow_{curr_time}.log')

def init():
    """Pre-processing code. Runs before main() to do any setup"""
    # Build log directory if necessary
    if not os.path.exists(f'./{LOG_DIR}'):
        os.mkdir(f'./{LOG_DIR}')
    if not os.path.exists(f'./{LOG_DIR}/training_logs'):
        os.mkdir(f'./{LOG_DIR}/training_logs')
    if not os.path.exists(f'./{LOG_DIR}/tensorflow.log'):
        f = open(f'./{LOG_DIR}/tensorflow.log', 'x')
        f.close()
    if not os.path.exists(f'./{LOG_DIR}/latest_run.log'):
        f = open(f'./{LOG_DIR}/latest_run.log', 'x')
        f.close()

    # Get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(f'./{LOG_DIR}/tensorflow.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)


    # Set up terminal logging
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s\n',
                    datefmt='%m-%d %H:%M',
                    filename=f'./{LOG_DIR}/latest_run.log',
                    filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

#########################################################   NOTES   ######################################################################

# Maybe we can identify certain operations?
# Identity is always identity matrix
# We can preserve arguments by passing forward args as identity matrix forward to next layer, same with operations we've gathered
# Lin. combinations look like [ ... a b c.... d] for something like ax_n + bx_n+1 + cx_n+2 + dx_m
# ... I can't figure out how to do multiplication of two variables in an array...

# Ideas:
#   - Maybe do one knowledge distillation step of teacher -> student to reduce size of the NN/make the FF step easier
#   - We can probably assume that over the domain we've learned, the teacher/student are "accurate enough" to throw random
#     vectors at it and trust the result
#   ✔ Memorize/store unit student NNs? We can build ASTs from these and pass the results of one into another
#   - We can also extend otherwise tricky/periodic functions with some pre/post-processing by doing things like % 2 pi
#   - We can also check for out of bounds/NaNs and throw them out of ASTs for things like lg(0), if the student/teacher return *something*
#       - Can we check for discontinuities in the NNs? I might play with this more...
#       - This might be useful: https://arxiv.org/pdf/2012.03016.pdf
#       - We need to build a NN with at least 2d + 1 hidden nodes in one of the layers though, it seems. Not a difficult thing to do

#  Unit NN's to hard-code:
#   - Addition: 0 hidden layers, weights = 1 bias = 0. Output has leaky ReLU w/ alpha = 1
#   - Subtraction: Basically the same, but w_1 = -1
#   - Abs: 1 hidden layer, 2 nodes. Each weight is 1 and negative 1 bias = 0, that's it.
#   - Negation: 0 hidden layers, leaky ReLU at output with alpha = 1, weight is -1
#   - Predecessor/successor: Addition with forced 1/-1 as second "hidden" input?
#   - Identity fn?: Just straight forward. Might be useful as a way to pass numbers straight-through, require LReLU w/ alpha = 1

# TODO List:
#   1. A metric that is able to quantify the error between the teacher network’s soft-label and
#   the student’s predicted label.
#   2. A training algorithm that leverages the above algorithm to train the student network.
#   3. An algorithm that is able to evaluate a cost metric for a string that represents the equation
#   a set of unit student NNs are modeling.
#   4. An algorithm to evaluate the associated Pareto score between the string cost and the
#   model accuracy.
#   5. An algorithm that generates a candidate Pareto frontier of student networks.
#   6. An algorithm that removes the Pareto dominated candidates from the frontier and repopoulates the frontier with a genetic algorithm to emulate the Pareto optimal candidates.
#   7. After satisfactory training, the best Pareto optimal candidate will be selected as the condensed model, and its string returned as the symbolic extraction.

#############################################################################################################################################


def main():
    start = timeit.default_timer()
    bf_unit_nns()
    stop = timeit.default_timer()

    minutes, seconds = divmod(stop-start, 60.0)
    hours, minutes = divmod(minutes, 60)
    print(f"Took {int(hours)}:{int(minutes)}:{seconds}")

if __name__ == "__main__":
    print('Initializing...', end='\r')
    init()
    main()