import tensorflow as tf
import numpy as np

class VGGBlock(tf.keras.layers.Layer):

    def __init__(self):
        super(VGGBlock, self).__init__()
        # self.input_size = input_size

        self.conv1 = tf.keras.layers.Conv2D(64, 257)
        self.conv2 = tf.keras.layers.Conv2D(128, 129)
        self.conv3 = tf.keras.layers.Conv2D(256, 65)
        self.shortcut = tf.keras.layers.Conv2D(256, 1, strides = 16)
        self.add = tf.keras.layers.Add()

        self.max_pool = tf.keras.layers.MaxPool2D((2, 2))

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool(x)
        shortcut = self.shortcut(input)
        output = self.add([x, shortcut])

        return output


class VGGModel(tf.keras.Model):

    def __init__(self):
        super(VGGModel, self).__init__()

        self.block = VGGBlock()
    def call(self, input):
        x = self.block(input)
        return x
    
    def build_graph(self, input_shape):

        x = tf.keras.layers.Input(shape = input_shape)
        return tf.keras.Model(inputs = [x], outputs = self.call(x))


if __name__ == "__main__":
    model = VGGModel()
    input = tf.random.uniform((1, 512, 512, 3))
    output = model(input)
    model.build_graph((512, 512, 3)).summary()
