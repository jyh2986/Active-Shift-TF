# Modify from https://github.com/soumith/convnet-benchmarks/blob/master/tensorflow/benchmark_alexnet.py

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


from datetime import datetime
import time
import math
import numpy as np
import tensorflow as tf
import lib.active_shift2d_op as active_shift2d_op


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 50,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('forward_only', False,
                            """Only run the forward pass.""")
tf.app.flags.DEFINE_boolean('forward_backward_only', True,
                            """Only run the forward-forward pass.""")
tf.app.flags.DEFINE_string('data_format', 'NCHW',
                           """The data format for Convnet operations.
                           Can be either NHWC or NCHW.
                           """)

parameters = []
timing_entries = []

def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    if not isinstance(target, list):
        target = [target]
    target_op = tf.group(*target)
    for i in range(FLAGS.num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target_op)
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print ('%s: step %d, duration = %.3f' %
                    (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / FLAGS.num_batches
    vr = total_duration_squared / FLAGS.num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
            (datetime.now(), info_string, FLAGS.num_batches, mn, sd))




def run_benchmark():
    global parameters
    timing_entries = []
    with tf.Graph().as_default():
        # Generate some dummy images.
        input_size = 64
        channel = 32

        if FLAGS.data_format == 'NCHW':
            input_shape = [FLAGS.batch_size, channel, input_size, input_size]
        else:
            input_shape = [FLAGS.batch_size, input_size, input_size, channel]
        shift_shape = [2, channel]
        images = tf.Variable(tf.random_normal(input_shape,
                                            dtype=tf.float32,
                                            stddev=1e-1))
        shift = tf.Variable(tf.random_normal(shift_shape,
                                            dtype=tf.float32,
                                            stddev=1e-1))
        parameters = [shift]

	last_layer = active_shift2d_op.active_shift2d_op(images, shift, strides=[1, 1, 1, 1], paddings=[0, 0, 0, 0])

        # Build an initialization operation.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session()
        sess.run(init)

        run_forward = True
        run_forward_backward = True
        if FLAGS.forward_only and FLAGS.forward_backward_only:
            raise ValueError("Cannot specify --forward_only and "
                            "--forward_backward_only at the same time.")
        if FLAGS.forward_only:
            run_forward_backward = False
        elif FLAGS.forward_backward_only:
            run_forward = False

        if run_forward:
            # Run the forward benchmark.
            timing_entries.append(time_tensorflow_run(sess, last_layer, "Forward"))

        if run_forward_backward:
            # Add a simple objective so we can calculate the backward pass.
            # objective = loss(last_layer, labels)
            loss = lambda x:tf.reduce_sum(x)
            objective = loss(last_layer)
            # Compute the gradient with respect to all the parameters.
            grad = tf.gradients(objective, parameters)
            # Run the backward benchmark.
            timing_entries.append(time_tensorflow_run(sess, grad, "Forward-backward"))

    # if FLAGS.csv_file:
    #     store_data_in_csv(timing_entries)


def main(_):
  run_benchmark()


if __name__ == '__main__':
  tf.app.run()
