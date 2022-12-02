"""Provides utilities for basic NN generation and plotting, compatible with the mystery_function class
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {‘0’, ‘1’, ‘2’}, stops tf from complaining about things it shouldn't be worried about

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import logging
from utils.conf import *
from utils.StudentTeacher import mystery_function

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
    plt.scatter(x, y1, label = "Ground Truth", color = "blue")
    plt.scatter(x, y2, label = "Predicted", color = "red")
    err = y2-y1
    plt.scatter(x, err, label = "Error", linestyle = "--", color = "green")


    plt.legend()
    plt.savefig(f'./{FIG_DIR}/{nn.name}.png')

def bf_unit_nns():
    """This is meant to iterate through every unit funciton we intend to use and find an ideal enough architecture for unit students"""

    binaries = {0: ['+', 'addition'], 1: ['-', 'subtraction']}
    unaries = {0: ['~', 'negation'], 1: ['A', 'abs'], 2: ['>', 'successor'], 3: ['<', 'predecessor']}
    ext = {0: ['L', 'log'], 1: ['S', 'sin'], 2: ['C', 'cos'], 3: ['s', 'arcsin'], 4: ['c', 'arccos'], 5: ['t', 'arctan'], 6: ['e^', 'exp']}


    if not os.path.exists(f'./{UNIT_DIR}'):
        os.mkdir(f'./{UNIT_DIR}')

    if not os.path.exists(f'./{LOG_DIR}/training_logs'):
        os.mkdir(f'./{LOG_DIR}/training_logs')

    # First, run over binary operators
    for i in range(len(binaries)):

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
            center = 2
        fn = mystery_function(f"x{{0}}x{{1}}{binaries[i][0]}", 2, gen_data=True, sample_size=100_000, center=center, scale=4)

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
    for i in range(len(unaries)):

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
    for i in range(len(ext)):

        # These next few lines simply adjust the domains for some fns to avoid getting undefined/NaN results during training
        center = 0
        scale = 10_000
        if i > 0 and i < 3:
            scale = 2 * math.pi
            center = math.pi
        if i > 2 and i < 5:
            scale = 2
        if i == 0:
            center = 50
            scale = 100
        if i == 6:
            scale = 10

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

# Reads in and returns our unit students as a tuple
def loadNNs(abs=True, atan=True, acos=True, asin=True, sin=True, cos=True, exp=True, log=True):
    NNs = []
    if abs:
        NNs.append(tf.keras.models.load_model(f'{UNIT_DIR}/abs'))
    if atan:
        NNs.append(tf.keras.models.load_model(f'{UNIT_DIR}/arctan'))
    if acos:
        NNs.append(tf.keras.models.load_model(f'{UNIT_DIR}/arccos'))
    if asin:
        NNs.append(tf.keras.models.load_model(f'{UNIT_DIR}/arcsin'))
    if sin:
        NNs.append(tf.keras.models.load_model(f'{UNIT_DIR}/sin'))
    if cos:
        NNs.append(tf.keras.models.load_model(f'{UNIT_DIR}/cos'))
    if exp:
        NNs.append(tf.keras.models.load_model(f'{UNIT_DIR}/exp'))
    if log:
        NNs.append(tf.keras.models.load_model(f'{UNIT_DIR}/log'))
    return NNs
