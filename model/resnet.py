from __future__ import division

import six
import tensorflow as tf

# based on code from https://github.com/raghakot/keras-resnet
from model.f1_metric import F1ScoreMetric

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3


def _fix_dims(rows, cols):
    if rows is None:
        return cols, cols
    elif cols is None:
        return rows, rows
    return rows, cols


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = tf.keras.layers.BatchNormalization(axis=CHANNEL_AXIS)(input)
    return tf.keras.layers.Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", tf.keras.regularizers.l2(1.e-4))

    def f(input):
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                      strides=strides, padding=padding,
                                      kernel_initializer=kernel_initializer,
                                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", tf.keras.regularizers.l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                      strides=strides, padding=padding,
                                      kernel_initializer=kernel_initializer,
                                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = tf.keras.backend.int_shape(input)
    residual_shape = tf.keras.backend.int_shape(residual)
    stride_width, stride_height = None, None
    if input_shape[ROW_AXIS] is not None:
        stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    if input_shape[COL_AXIS] is not None:
        stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    stride_width, stride_height = _fix_dims(stride_width, stride_height)
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = tf.keras.layers.Conv2D(filters=residual_shape[CHANNEL_AXIS],
                                          kernel_size=(1, 1),
                                          strides=(stride_width, stride_height),
                                          padding="valid",
                                          kernel_initializer="he_normal",
                                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))(input)

    return tf.keras.layers.Add()([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    print(f'building residual block with {filters} filters')
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    print(f'building basic block with {filters} filters')
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3),
                                           strides=init_strides,
                                           padding="same",
                                           kernel_initializer="he_normal",
                                           kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    print(f'building bottleneck with 4x{filters} filters')

    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1),
                                              strides=init_strides,
                                              padding="same",
                                              kernel_initializer="he_normal",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        if 'basic' == identifier:
            return basic_block
        elif 'bottleneck' == identifier:
            return bottleneck
        else:
            raise ValueError('Invalid {}'.format(identifier))
    return identifier


def _build_resnet_core(input, block_fn, repetitions, initial_filters=64, filter_scaler=2, maximum_filters=999999999):
    """Builds the blocks of layers making up the middle of a ResNet architecture, allowing the construction of
    custom ResNet-like architectures. This builds only the ResNet blocks, using a supplied input and returning the
    output of the block layers.

    Args:
        input: Input to the ResNet blocks
        block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
            The original paper used basic_block for layers < 50
        repetitions: Number of repetitions of various block units.
            At each block unit, the number of filters are scaled upward (to the specified maximum) and input size is halved
        initial_filters: Number of filters for first block.
            The default is 64, matching the standard ResNet architecture.
        filter_scaler: Multiplier for filter count in successive layers.
            The default is 2, matching the standard ResNet architecture. Actual filter counts are rounded up to a
            multiple of 16 at each layer.
        maximum_filters: Maximum number of filters.
            In the standard ResNet filters are doubled for each block. This allows the number of filters to be capped
            to limit the complexity of the model.

    Returns:
        (Final output from ResNet layers, list of intermediate outputs from blocks)
    """

    # Load function from str if needed.
    block_fn = _get_block(block_fn)

    block = input
    filters = initial_filters
    intermediates = []
    for i, r in enumerate(repetitions):
        block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
        intermediates.append(block)
        filters = min(maximum_filters, -(-int(filters * filter_scaler) // 16) * 16)

    # Last activation
    return _bn_relu(block), intermediates
