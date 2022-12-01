"""Rough Draft for Interpretable Student-Teacher NNs on SR

Authors: Brenton Candelaria and Sophia Novo-Gradac

Testing ground for code, has some solid work done in advance to make more tedious work easier in the future

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {‘0’, ‘1’, ‘2’}, stops tf from complaining about things it shouldn't be worried about

from utils import *
import numpy as np
import tensorflow as tf
import logging
import timeit # Just so I know how much time I've wasted
from datetime import datetime


def init():
    """Pre-processing code. Runs before main() to do any setup"""

    # Save old logs
    curr_time =datetime.now().strftime('%b-%d-%Y_%H%M%S')
    if False and os.path.exists(f'./{LOG_DIR}') and os.path.getsize(f'./{LOG_DIR}/latest_run.log') > 10:
        os.rename(f'./{LOG_DIR}/training_logs/', f'./{LOG_DIR}/training_logs_{curr_time}/')
        os.rename(f'./{LOG_DIR}/latest_run.log', f'./{LOG_DIR}/latest_run_{curr_time}.log')
        os.rename(f'./{LOG_DIR}/tensorflow.log', f'./{LOG_DIR}/tensorflow_{curr_time}.log')


    # Build log directory and logs if necessary (likely always going to happen)
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

def main():
    start = timeit.default_timer()
    # bf_unit_nns()

    models = loadNNs(abs=False, acos=False, asin=False, atan=False, exp=False, log=False)
    for m in models:
        for l in m.layers:
            l.trainable = False
    logging.info(f'Loaded unit student models')

    # fn = mystery_function("0>>x{0}0>>x{1}^*/", 2, True, scale = 10)
    fn = mystery_function("x{0}S", 1, True, scale=10, sample_size=1_000)
    # fn = mystery_function("x{0}x{1}*", 2, True, scale=10, sample_size=10_000)
    logging.info(f'Mystery function generated')


    # Let's just go with a random teacher architecture and use it to work with for now
    teacher = getNN(10, 256, fn.shape, name='Teacher')
    logging.info(f'Teacher model generated')
    t_params = np.sum([np.prod(v.get_shape().as_list()) for v in teacher.trainable_variables])
    logging.info(f'Teacher has {t_params} trainable params')
    
    teacher.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error') # Compile it
    teacher.fit(fn.x_train, fn.y_train, epochs=5) # Train
    teacher.evaluate(fn.x_test, fn.y_test)

    logging.info(f"Beginning knowledge distillation step")

    for i in range(6):
        student = construct_student(fn.shape, i, "Student", models)
        logging.info(f'Student model {i} generated')
        s_params = np.sum([np.prod(v.get_shape().as_list()) for v in student.trainable_variables])
        logging.info(f'Student has {s_params} trainable params\nCR: \
                {s_params/t_params:.5f}')
        distiller = Distiller(student=student, teacher=teacher)
        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.MeanSquaredError()],
            student_loss_fn=tf.keras.losses.MeanSquaredError(),
            distillation_loss_fn=tf.keras.losses.MeanSquaredError(),
            alpha=0.1,
            temperature=10,
        )

        # Distill teacher to student
        history = distiller.fit(fn.x_train, fn.y_train, epochs=EPOCHS)

        # Evaluate student on test dataset
        distiller.evaluate(fn.x_test, fn.y_test)
    stop = timeit.default_timer()

    minutes, seconds = divmod(stop-start, 60.0)
    hours, minutes = divmod(minutes, 60)
    print(f"Took {int(hours):2d}:{int(minutes):2d}:{seconds:2.3f}")

if __name__ == "__main__":
    print('Initializing...', end='\r')
    init()
    main()