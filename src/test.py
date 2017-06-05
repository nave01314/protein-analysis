# This file is for experimentation with TensorFlow

import os
import tensorflow as tf
import numpy as np


def test_learning(primary, secondary, v_primary, v_secondary):
    # Don't remove this, I need it to mitigate tf build warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    x = tf.placeholder(tf.float32, [None, len(primary[0])])
    W = tf.Variable(tf.zeros([len(primary[0]), len(secondary[0])]))
    b = tf.Variable(tf.zeros([len(secondary[0])]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, len(secondary[0])])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = primary, secondary
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return sess.run(accuracy, feed_dict={x: v_primary, y_: v_secondary})
