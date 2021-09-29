import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

print(tf.__version__)

mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


