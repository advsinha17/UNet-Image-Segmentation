import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

CWD = os.path.dirname(__file__)
        
class DoubleConv(tf.keras.layers.Layer):

    """

    Class that returns the double convultional layers required for each block of a UNet.

    Args:
        n_filters: Number of output features in the convolution

    """
    def __init__(self, n_filters):
        super(DoubleConv, self).__init__()

        # conv layer
        self.conv = tf.keras.layers.Conv2D(
                    n_filters, 
                    kernel_size=(3, 3), 
                    padding='same',
                    kernel_initializer='he_normal'
                )
        
        # batch norm layer
        self.bn   = tf.keras.layers.BatchNormalization()
    
    def call(self, input_tensor, training=False):

        """

        Returns the output of two Convolution -> Batch Normalization -> ReLU units given the input tensor.

        Args:
            input_tensor: Input tensor for the DoubleConv unit.
            training: True if model is being trained, False otherwise.

        Returns:
            x: The output of the DoubleConv unit.

        """
        
        # First DoubleConv unit, comprises of Convolution layer -> Batch Normalization layer -> ReLU activation
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        # Second DoubleConv unit
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        return x
      
class DownsamplingBlock(tf.keras.layers.Layer):

    """

    Class that defines one Downsampling/Encoding unit of the UNet.

    Args:
        n_filters: Number of output filters for the DoubleConv unit of each downsampling unit.
        dropout_prob: Dropout probability for the dropout layer.
    
    """

    def __init__(self,   
                 n_filters, 
                 dropout_prob = 0
                ):
        super(DownsamplingBlock, self).__init__()

        # DoubleConv layer
        self.double_conv = DoubleConv(n_filters)

        # Dropout layer
        self.Dropout = tf.keras.layers.Dropout(dropout_prob)

        # MaxPooling layer
        self.MaxPooling = tf.keras.layers.MaxPooling2D(pool_size = (2, 2))

    def call(self, input_tensor, max_pooling = True, training = False):
        """

        Returns the output for the next downsampling unit and the information to be passed to corresponding upsampling unit
        through a skip connection.

        Args:
            input_tensor: Inpur tensor to the downsampling unit.
            max_pooling: If max pooling is to be done before passing to the next downsampling unit. Not done for the last
            downsampling unit.
            training: True if model is being trained, False otherwise.
        
        Returns:
            next_layer: Output passed to the next downsampling unit.
            skip_connection: Output passed to corresponding upsampling unit through a skip connection.

        """

        # DoubleConv unit
        x = self.double_conv(input_tensor)

        # Dropout layer
        x = self.Dropout(x, training = training)

        # Max pooling layer
        if max_pooling:
            next_layer = self.MaxPooling(x)

        else: 
            next_layer = x

        skip_connection = x
        
        return next_layer, skip_connection

class UpsamplingBlock(tf.keras.layers.Layer):

    """

    Class that defines one Upsampling/Decoding unit of the UNet.

    Args:
        n_filters: Number of output filters for the DoubleConv unit of each downsampling unit.
    
    """

    def __init__(self, n_filters = 32):

        super(UpsamplingBlock, self).__init__()

        # DoubleConv layer
        self.double_conv = DoubleConv(n_filters)

        # Transpose convolution layer
        self.transpose = tf.keras.layers.Conv2DTranspose(n_filters,
                                                         kernel_size = (3, 3),
                                                         strides = 2,
                                                         padding = 'same')
        
        # Concatenation layer
        self.concatenate = tf.keras.layers.Concatenate(axis = 3)

    def call(self, input_tensor, skip_input, training = False):

        """

        Returns output for the next upsampling unit.

        Args:
            input_tensor: Inpur tensor to the unpsampling unit.
            skip_input: Input from corresponding downsampling block passed through a skip connection.
            training: True if model is being trained, False otherwise.
        
        Returns:
            x: Output of each unsampling unit.

        """
        
        # Transpose Convolution.
        up = self.transpose(input_tensor)

        # Concatenation of transpose convolution output with skip_input before passing to DoubleConv layer.
        x = self.concatenate([up, skip_input])

        # DoubleConv layer.
        x = self.double_conv(x, training = training)

        return x
    
class UNet(tf.keras.Model):

    """
    
    Class to return the UNet model, compiling the Downsampling and Upsampling blocks.

    Args:
        n_classes: Number of classes in encoded image to predict.
        n_filters: number of output filtes for the DoubleConv units.
    
    """

    def __init__(self, n_classes = 3, n_filters = 32):
         
        super(UNet, self).__init__()

        # Downsampling blocks
        self.downsampling_block1 = DownsamplingBlock(n_filters)
        self.downsampling_block2 = DownsamplingBlock(n_filters * 2)
        self.downsampling_block3 = DownsamplingBlock(n_filters * 4)
        self.downsampling_block4 = DownsamplingBlock(n_filters * 8, dropout_prob = 0.3)
        self.downsampling_block5 = DownsamplingBlock(n_filters * 16, dropout_prob = 0.3)
    
        # Upsampling blocks
        self.upsampling_block1 = UpsamplingBlock(n_filters * 8)
        self.upsampling_block2 = UpsamplingBlock(n_filters * 4)
        self.upsampling_block3 = UpsamplingBlock(n_filters * 2)
        self.upsampling_block4 = UpsamplingBlock(n_filters)

        # conv layer
        self.conv = tf.keras.layers.Conv2D(n_filters,
                                           kernel_size = (3, 3),
                                           padding = 'same',
                                           kernel_initializer = 'he_normal')
        
        # BatchNorm layer
        self.bn = tf.keras.layers.BatchNormalization()

        # Output conv layer
        self.output_conv = tf.keras.layers.Conv2D(n_classes, 1, padding = 'same')

    def call(self, input_tensor, training = False):

        """
        Compiles and returns the output of the UNet model.

        Args:
            input_tensor: Input tensor to the UNet model.
            training: True if model is being trained, False otherwise.

        Returns:
            outputs: Output of the UNet model. For an input image of size (128 x 128 x 3), outputs image of size
            (128 x 128 x 3).
        
        """

        # Downsampling blocks
        down1 = self.downsampling_block1(input_tensor, training = training)
        down2 = self.downsampling_block2(down1[0], training = training)
        down3 = self.downsampling_block3(down2[0], training = training)
        down4 = self.downsampling_block4(down3[0], training = training)
        down5 = self.downsampling_block5(down4[0], max_pooling = False, training = training)

        # Upsampling blocks
        up1 = self.upsampling_block1(down5[0], down4[1], training = training) 
        up2 = self.upsampling_block2(up1, down3[1], training = training)
        up3 = self.upsampling_block3(up2, down2[1], training = training)
        up4 = self.upsampling_block4(up3, down1[1], training = training)

        # conv layer
        x = self.conv(up4)

        # BatchNorm layer
        x = self.bn(x, training = training)

        # ReLU layer
        x = tf.nn.relu(x)

        # Ouput conv layer
        outputs = self.output_conv(x)

        return outputs
    
    def build_graph(self, input_shape):

        """
        Builds the model given input shape.

        Args:
            input_shape: Shape of input tensor.
        
        Returns:
            Model built given the input shape.

        """
        x = tf.keras.layers.Input(shape = input_shape)
        return tf.keras.Model(inputs = [x], outputs = self.call(x))

def test():
    input_size = (128, 128, 3)
    model = UNet()
    model.build_graph(input_size).summary()
    tf.keras.utils.plot_model(model.build_graph(input_size), to_file=os.path.join(CWD, 'model.png'), show_shapes=True)

if __name__ == "__main__":
    test()