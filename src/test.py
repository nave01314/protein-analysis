# This file is for experimentation with TensorFlow

import os
import tensorflow as tf
import numpy as np


def test_learning():
    # Don't remove this, I need it to mitigate tf build warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Model parameters
    a = tf.Variable([1.], tf.float64)
    b = tf.Variable([1.], tf.float64)
    c = tf.Variable([1.], tf.float64)
    d = tf.Variable([1.], tf.float64)
    # Model input and output
    x = tf.placeholder(tf.float32)
    model = a * x ** 3 + b * x ** 2 + c * x + d
    y = tf.placeholder(tf.float32)
    # Loss
    squared_deltas = tf.square(model-y)
    loss = tf.reduce_mean(squared_deltas)
    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    # Training Data
    x_train = [-2, -1, 0, 1, 2]
    y_train = [-2, -1, 0, 1, 2]
    # Training Loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
        curr_a, curr_b, curr_c, curr_d = sess.run([a, b, c, d], {x: x_train, y: y_train})
        print("Formula: %s x^3 + %s x^2 + %s x + %s" % (curr_a, curr_b, curr_c, curr_d))
        sess.run([train], {x: x_train, y: y_train})
    # Evaluate Training Accuracy
    curr_a, curr_b, curr_c, curr_d = sess.run([a, b, c, d], {x: x_train, y: y_train})
    print("Formula: %s x^3 + %s x^2 + %s x + %s" % (np.round(curr_a), np.round(curr_b), np.round(curr_c), np.round(curr_d)))
