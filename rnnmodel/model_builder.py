
import tensorflow as tf
from tensorflow.keras import layers as layers


def build_model(input_dims, num_classes, custom_objects=None, **parms):
    inputs = [layers.Input(shape=d) for d in input_dims]
    print(f'created model input layer with shapes {input_dims}')
    if len(inputs) == 1:
        input = inputs[0]
    else:
        input = layers.Concatenate()(inputs)
    x = input
    x = layers.Masking()(x)
    layer_defs = parms.get('layers')
    before_rnn = True
    for layer_def in layer_defs:
        layer_type = layer_def['type']
        node_count = layer_def['count']
        batch_norm = layer_def.get('batch_norm', False)
        dropout_rate = layer_def.get('dropout_rate')
        if layer_type == 'dense':
            activation = layer_def.get('activation', 'relu')
            if before_rnn:
                if batch_norm or dropout_rate is not None:
                    seq = tf.keras.models.Sequential()
                    if dropout_rate is not None:
                        print(f'WARNING: Dropout in TimeDistributed doesn\'t generally work well')
                        seq.add(layers.Dropout(rate=dropout_rate))
                    if batch_norm:
                        seq.add(layers.Dense(node_count, kernel_initializer='he_normal'))
                        seq.add(layers.BatchNormalization())
                        seq.add(layers.Activation('activation'))
                    else:
                        seq.add(layers.Dense(node_count, kernel_initializer='he_normal', activation=activation))
                    x = layers.TimeDistributed(seq)(x)
                else:
                    x = layers.TimeDistributed(layers.Dense(node_count, kernel_initializer='he_normal', activation=activation))(x)
            else:
                if dropout_rate is not None:
                    x = layers.Dropout(rate=dropout_rate)(x)
                if batch_norm:
                    x = layers.Dense(node_count, kernel_initializer='he_normal')(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.Activation('relu')(x)
                else:
                    x = layers.Dense(node_count, kernel_initializer='he_normal', activation=activation)(x)
        else:
            before_rnn = False
            return_sequences = layer_def.get('return_sequences', False)
            if layer_type == 'gru':
                x = layers.GRU(node_count, return_sequences=return_sequences)(x)
            elif layer_type == 'lstm':
                x = layers.LSTM(node_count, return_sequences=return_sequences)(x)
            else:
                raise ValueError(f'Unsupported layer type "{layer_type}"')
            if batch_norm:
                x = layers.BatchNormalization()(x)
    classifier_dropout = parms.get('classifier_dropout')
    if classifier_dropout is not None:
        x = layers.Dropout(rate=classifier_dropout)(x)
    x = tf.keras.layers.Dense(units=num_classes, kernel_initializer='he_normal', name='class-dense')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    activation = parms["classifier_activation"]
    class_out = tf.keras.layers.Activation(activation, name='class')(x)
    optimizer = tf.keras.optimizers.deserialize(parms['optimizer'])
    loss = tf.keras.losses.deserialize(parms['loss'])
    metrics = [tf.keras.metrics.deserialize(c, custom_objects=custom_objects) for c in parms['metrics']]
    model = tf.keras.models.Model(inputs=inputs, outputs=class_out)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

