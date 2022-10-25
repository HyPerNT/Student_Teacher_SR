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



class mystery_function():
    def __init__(self, fn, dim, sample_size=DEFAULT_SAMPLE_SIZE, scale=DEFAULT_SCALE, test_size=DEFAULT_TEST_SIZE, noisy=False, noise_factor=DEFAULT_NOISE):
        self.fn = fn
        self.dim = dim
        self.gen_data(fn, sample_size=sample_size, scale=scale, test_size=test_size, noisy=noisy, noise_factor=noise_factor)

    def gen_data(self, fn, sample_size=DEFAULT_SAMPLE_SIZE, scale=DEFAULT_SCALE, test_size=DEFAULT_TEST_SIZE, noisy=False, noise_factor=DEFAULT_NOISE):
        r = np.random.rand(int(sample_size*(1-test_size)), self.dim) * scale
        self.x_train = r
        self.y_train = Eval(fn, r)[:,-1]

        r = np.random.rand(int(sample_size*test_size), self.dim) * scale
        self.x_test = r
        self.y_test = Eval(fn, r)[:,-1]
        if noisy:
            self.x_train = self.x_train + noise_factor * np.random.rand(int(sample_size * (1 - test_size)), self.dim) - 0.5      
            for i in self.y_train:
                i = i + noise_factor * np.random.rand((0)) - 0.5

            self.x_test = self.x_test + noise_factor * np.random.rand(int(sample_size * test_size), self.dim) - 0.5 
            for i in self.y_test:
                i = i + noise_factor * np.random.rand((0)) - 0.5


def getNN(layers, nodes, in_shape, rate=0.01, name='nn'):
    model = tf.keras.Sequential([tf.keras.Input(shape=in_shape)],name=name)
    for _ in range(layers):
        model.add(tf.keras.layers.Dense(nodes, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=rate))
    model.add(tf.keras.layers.Dense(1))
    return model

def iterateNN(in_shape, fn, layer_range=10, node_range=512, by=2, method="*", rate=0.01, name='nn'):
    best_nn_loss = -1
    best_ij = [-1,-1]
    best_confs = []
    name_base = name
    best_model = None
    history = []
    for i in range(layer_range):
        j = 1
        while j <= node_range:
            name = name_base + f"_{i}-layers_{j}-nodes"
            print(f'\n\nTraining {name}...\n{(len(name)+12)*"-"}')
            model = getNN(i, j, in_shape, rate, name)
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
            train = model.fit(fn.x_train, fn.y_train, epochs=EPOCHS)
            predict = model.evaluate(fn.x_test, fn.y_test)
            train = min(train.history['loss'])
            model.summary()
            history.append([[i,j], train+predict, model.count_params()])
            if train+predict < best_nn_loss or best_nn_loss == -1:
                best_nn_loss = train+predict
                best_ij = [i,j]
                best_confs.append([best_ij, best_nn_loss])
                best_model = model
                print(f"{(len(best_ij)+16)*'-'}\nNew Best: {best_ij}\nScore: {best_nn_loss:1.4f}\n{(len(best_ij)+16)*'-'}")
            if method=="*":
                j*=by
            elif method=="+":
                j+=by
    print(f"Best Configuration: {best_ij[0]} layers and {best_ij[1]} nodes/layer, loss of {best_nn_loss:1.4f}")
    return best_confs, best_model, history

def main():
    start = timeit.default_timer()
    fn1 = mystery_function("0>>x{0}0>>x{1}^*/", 2, sample_size=10_000, scale=10, noisy=True)
    # sin = mystery_function("x{0}S", 1, noisy=True)

    # All records like [  [i,j], loss,  ] or [  [i,j], loss, params  ] 

    records, best_nn, all_models= iterateNN((2,), fn1, layer_range=10, name='KE_nn')
    stop = timeit.default_timer()
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