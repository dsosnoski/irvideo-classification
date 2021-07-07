from __future__ import division

import efficientnet.tfkeras as efn
import tensorflow as tf

from model.f1_metric import F1ScoreMetric
from model.resnet import _conv_bn_relu, _build_resnet_core, _fix_dims
from model.model_certainty import ModelWithCertainty


def _build_pool(input, **parms):
    pool_type = parms['type']
    poolparms = parms.copy()
    del poolparms['type']
    nested = poolparms['params']
    poolparms.update(nested)
    del poolparms['params']
    if pool_type == 'max':
        return tf.keras.layers.MaxPooling2D(**poolparms)(input)
    elif pool_type == 'avg':
        return tf.keras.layers.AveragePooling2D(**poolparms)(input)
    else:
        raise ValueError(f'Unknown pool type {pool_type}')


def _dense_norm_relu(n, x):
    x = tf.keras.layers.Dense(n, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)


def _build_band(input, **parms):
    x = input
    if 'initial_conv' in parms and parms['initial_conv']:
        x = _conv_bn_relu(**parms['initial_conv'])(x)
    if 'initial_pool' in parms and parms['initial_pool']:
        x = _build_pool(x, **parms['initial_pool'])
    feature_extraction = parms['feature_extraction']
    feature_parms = feature_extraction['params']
    if feature_extraction['type'] == 'resnet':
        x, intermediate_outputs = _build_resnet_core(x, **feature_parms)
    elif feature_extraction['type'] == 'efficient':
        input_shape = tf.keras.backend.int_shape(x)[1:]
        feature_extractor = efn.EfficientNetB1(input_shape=input_shape, include_top=False, **feature_parms)
        x = feature_extractor(x)
        intermediate_outputs = []
    else:
        raise ValueError(f'Unknown feature_extraction type {feature_extraction["type"]}')
    if 'gru-classifier' in parms:
        gru_parms = parms['gru-classifier']
        xalt = x
        block_shape = tf.keras.backend.int_shape(x)
        conv_parms = {}
        conv_parms['filters'] = block_shape[3]
        conv_parms['kernel_size'] = [block_shape[1], 2]
        conv_parms['strides'] = [block_shape[1], 1]
        if 'initial-conv' in gru_parms:
            conv_parms.update(gru_parms['initial-conv'])
        x = _conv_bn_relu(**conv_parms)(x)
        block_shape = tf.keras.backend.int_shape(x)
        if block_shape[1] > 1:
            x = _build_pool(x, type='max', params={'pool_size': [block_shape[1], 1], 'strides': [block_shape[1], 1],
                                                   'padding': 'valid'})
            block_shape = tf.keras.backend.int_shape(x)
        x = tf.keras.layers.Reshape((-1, block_shape[3]))(x)
        counts = gru_parms['counts']
        for i in range(len(counts)):
            x = tf.keras.layers.GRU(counts[i], return_sequences=(i < len(counts) - 1))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        inputs = [x]
        if 'bypass' in gru_parms:
            bypass_parms = gru_parms['bypass']
            form = bypass_parms['form']
            if form == 'convpool':
                xalt = _conv_bn_relu(**bypass_parms['config'])(xalt)
            if form == 'convpool' or form == 'maxpool':
                xalt = tf.keras.layers.GlobalMaxPooling2D()(xalt)
                xalt = tf.keras.layers.Flatten()(xalt)
            else:
                raise ValueError(f'Unknown bypass form {form}')
            inputs.append(xalt)
        if 'intermediates' in gru_parms:
            counts = gru_parms['intermediates']
            for index in range(len(counts)):
                inter = intermediate_outputs[index]
                block_shape = tf.keras.backend.int_shape(inter)
                conv_parms = {
                    'filters': counts[index],
                    'kernel_size': [block_shape[1], 3],
                    'strides': [block_shape[1], 2]
                }
                inter = _conv_bn_relu(**conv_parms)(inter)
                block_shape = tf.keras.backend.int_shape(inter)
                inter = tf.keras.layers.Reshape((-1, block_shape[3]))(inter)
                inter = tf.keras.layers.GRU(counts[index])(inter)
                inter = tf.keras.layers.BatchNormalization()(inter)
                inputs.append(inter)
        if len(inputs) > 1:
            x = tf.keras.layers.Concatenate()(inputs)
    else:
        if 'final_conv' in parms:
            conv_parms = parms['final_conv']
            if 'filters' not in conv_parms:
                block_shape = tf.keras.backend.int_shape(x)
                conv_parms['filters'] = block_shape[3]
            x = _conv_bn_relu(**conv_parms)(x)
        if 'final_pool' in parms:
            block_shape = tf.keras.backend.int_shape(x)
            if block_shape[1] is None or block_shape[2] is None:
                x = tf.keras.layers.GlobalMaxPool2D()(x)
            else:
                w, h = _fix_dims(block_shape[1], block_shape[2])
                x = _build_pool(x, pool_size=(w, h), strides=(1, 1), **parms['final_pool'])
        if 'dropout' in parms:
            x = tf.keras.layers.Dropout(parms['dropout'])(x)
        if len(tf.keras.backend.int_shape(x)) > 1:
            x = tf.keras.layers.Flatten()(x)
    return x, intermediate_outputs


class ModelBuilder(object):

    @staticmethod
    def build_model(input_dims, num_classes, certainty_loss_weight=None, **parms):
        input = tf.keras.layers.Input(shape=input_dims)
        print(f'created model input layer with shape {input_dims}')
        x, intermediate_outputs = _build_band(input, **parms)
        if 'dense-classifier' in parms:
            for n in parms['dense-classifier']:
                x = tf.keras.layers.Dropout(rate=.5)(x)
                x = _dense_norm_relu(n, x)
        activation = parms["classifier_activation"]
        class_dense = tf.keras.layers.Dense(units=num_classes, kernel_initializer='he_normal', name='class-dense')(x)
        class_out = tf.keras.layers.Activation(activation, name='class')(class_dense)
        optimizer = tf.keras.optimizers.deserialize(parms['optimizer'])
        loss = tf.keras.losses.deserialize(parms['loss'])
        metrics = [tf.keras.metrics.deserialize(c, custom_objects={'F1ScoreMetric': F1ScoreMetric}) for c in parms['metrics']]
        if certainty_loss_weight is not None:
            interx1 = tf.keras.layers.Flatten()(intermediate_outputs[-1])
            interx1 = tf.keras.layers.Dropout(rate=.5)(interx1)
            interx1 = _dense_norm_relu(16, interx1)
            interx2 = tf.keras.layers.Dropout(rate=.5)(class_dense)
            interx2 = tf.keras.layers.Activation('relu')(interx2)
            interx = tf.keras.layers.Concatenate()([interx1, interx2])
            certainty = tf.keras.layers.Dense(1, kernel_initializer='he_normal', activation='sigmoid', name='certainty')(interx)
            model = ModelWithCertainty(inputs=[input], outputs=[class_out, certainty])
            model.compile(optimizer=optimizer, loss=[loss, tf.keras.losses.BinaryCrossentropy()], loss_weights=[1, certainty_loss_weight], metrics=[metrics, tf.keras.metrics.BinaryCrossentropy()])
        else:
            model = tf.keras.models.Model(inputs=[input], outputs=class_out)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
