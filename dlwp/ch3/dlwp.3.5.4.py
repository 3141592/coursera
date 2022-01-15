import tensorflow as tf
import numpy as np

# Listing 3.17 Creating the linear classifier variables
input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
r = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

# Listing 3.18 The forward pass function
def model(inputs):
    return tf.matmul(inputs, W) + b



