import tensorflow as tf
from tensorflow.keras import layers

class MNISTConv(tf.keras.Model):
    """CNN that matches the Baseline MNIST architecture."""
    def __init__(self):
        super(MNISTConv, self).__init__()
        self.conv1 = layers.Conv2D(16, kernel_size=5, activation='relu')
        self.pool1 = layers.AveragePooling2D(pool_size=2, strides=2)

        self.conv2 = layers.Conv2D(32, kernel_size=5, activation='relu')
        self.pool2 = layers.AveragePooling2D(pool_size=2, strides=2)

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation='relu')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

class SingleDigit(tf.keras.Model):
    """Final classifier head for the Baseline MNIST model."""
    def __init__(self, inputs_as_a_list=False):
        super(SingleDigit, self).__init__()
        self.mnistconv = MNISTConv()
        self.fc2 = layers.Dense(10, activation='softmax')
        self.inputs_as_a_list = inputs_as_a_list

    def call(self, inputs):
        x = inputs if not self.inputs_as_a_list else inputs[0]
        x = self.mnistconv(x)
        x = self.fc2(x)
        return x
