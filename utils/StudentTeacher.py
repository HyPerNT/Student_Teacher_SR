"""Classes and definitions for custom objects to allow for Student-Teacher Network generation"""


import tensorflow as tf
import numpy as np
from utils.conf import *
from utils.eval import Eval

class mystery_function():
    """A class to model a given fn, also handles creation/storage of data to train/test on"""
    def __init__(self, fn, dim, gen_data=False, sample_size=DEFAULT_SAMPLE_SIZE, scale=DEFAULT_SCALE, center=DEFAULT_CENTER, test_size=DEFAULT_TEST_SIZE, noisy=False, noise_factor=DEFAULT_NOISE, outlier_rate=DEFAULT_OUTLIER):
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
            self.gen_data(sample_size=sample_size, scale=scale, test_size=test_size, center=center, noisy=noisy, noise_factor=noise_factor, outlier_rate=outlier_rate)

    def gen_data(self, sample_size=DEFAULT_SAMPLE_SIZE, scale=DEFAULT_SCALE, center=DEFAULT_CENTER, test_size=DEFAULT_TEST_SIZE, noisy=False, noise_factor=DEFAULT_NOISE, sorted=False, outlier_rate=DEFAULT_OUTLIER):
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
        
        if outlier_rate > 0:
            mask = np.random.choice([0,1],size=self.x_train.shape, p=((1-outlier_rate), outlier_rate)).astype(np.bool)
            r = np.random.rand(*self.x_train.shape)*np.max(self.x_train)
            self.x_train[mask] = r[mask]

            mask = np.random.choice([0,1],size=self.y_train.shape, p=((1-outlier_rate), outlier_rate)).astype(np.bool)
            r = np.random.rand(*self.y_train.shape)*np.max(self.y_train)
            self.y_train[mask] = r[mask]

# Distiller class for Student-Teacher capability, ripped from a TF Docs page
class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    teacher_predictions / self.temperature,
                    student_predictions / self.temperature,
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"s_loss": student_loss, "d_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

class FnLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, NNs, repeats=1):
        super(FnLayer, self).__init__()
        self.units = input_dim + repeats*len(NNs)
        self.NNs = NNs
        self.repeats = repeats

    def build(self, input_shape):
        # Our w and b matrices shouldn't do anything, the Dense layers do the work for us
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True
        )
    def call(self, inputs): 
        """This is broken"""
        # Get our input from the dense layer
        mat = tf.matmul(inputs, self.w) + self.b

        # Init result so it has the correct shape
        result = self.NNs[0](mat[0])[0]

        # Iterate over unit students
        for i in range(1, self.repeats*len(self.NNs)):
            # Ask them for an answer, add it to the result tensor
            k = i % len(self.NNs)
            arg = mat[k]
            if k > 3 and k < 5:
                arg %= 2 * np.pi
            ans = self.NNs[k](arg)[k]

            result = tf.concat((result, ans), axis=0)

        # This basically makes the shape correct, I think? But it breaks our training apparently

        # Make a tensor of zeroes to pad with, pad result so the top rows = the input (for passing past layers forward),
        # bottom rows = the answers we've just obtained from the unit students on this layer
        sh = result.shape[0]
        padding = tf.constant([[0,0], [self.units - sh, 0]])
        result = tf.reshape(result, [-1, sh])
        result = tf.pad(result, padding, "CONSTANT")

        # Make an identity matrix with some 0s. Multiply by the original input to drop the bottom rows (unnecessary,
        # they should be nonsense anyway for our purposes)
        diag = [1.0 if i < inputs.shape[1] else 0.0 for i in range(self.units)]
        mask = tf.linalg.tensor_diag(diag)
        mask = tf.matmul(mat, mask)
        result = tf.reshape(result, [-1, result.shape[1]])

        # Return a matrix like [in[0], in[1], ..., in[n], unit[0], unit[1], ..., unit[m]
        return result + mask

# Builds a student of desired depth, depth is proportional to the nesting of operations/height of the AST to read
def construct_student(in_shape, layers, name, NNs):
    # Make template NN
    model = tf.keras.Sequential([tf.keras.Input(shape=in_shape)],name=name)

    # Iteratively add layers
    for i in range(layers):
        model.add(FnLayer(in_shape[0] if i < 1 else model.layers[i-1].output_shape[1], NNs))

    # Add final layer for regression
    model.add(tf.keras.layers.Dense(1))

    return model
