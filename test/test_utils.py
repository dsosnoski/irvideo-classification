import os

import numpy as np
import tensorflow as tf

from model.f1_metric import F1ScoreMetric
from model.model_certainty import ModelWithCertainty
from support.data_model import TAG_CLASS_MAP


def flatten_tag_tracks(tag_tracks):
    test_tracks = []
    for tag in tag_tracks.keys():
        if tag in TAG_CLASS_MAP:
            tracks = tag_tracks[tag]
            test_tracks = test_tracks + tracks
        else:
            print(f'skipping {len(tag_tracks[tag])} tracks with unsupported tag {tag}')
    return test_tracks


def load_tracks(tracks, data_path):
    with open(data_path, 'rb') as f:
        for track in tracks:
            assert track.data is None
            f.seek(track.offset)
            track.data = np.load(f)
            assert track.frame_count == len(track.data)


def prep_track(track, input_shape):
    frame_samples = []
    for frame_index in range(track.frame_count):
        adjust = track.data[frame_index].astype(np.float32)
        h, w, _ = input_shape
        r0, c0 = 0, 0
        r1, c1 = h, w
        if adjust.shape[0] > r1:
            offset = (adjust.shape[0] - r1) // 2
            r0 += offset
            r1 += offset
        if adjust.shape[1] > c1:
            offset = (adjust.shape[1] - c1) // 2
            c0 += offset
            c1 += offset
        adjust = adjust[r0:r1, c0:c1]
        if input_shape[2] == 1:
            frame_samples.append(adjust)
        else:
            frame_samples.append(np.repeat(adjust[..., np.newaxis], input_shape[2], -1))
    return np.array(frame_samples)


def convert_remote_path(path):
    if path.endswith('.index'):
        path = path[:-6]
    if path.startswith('fish://dennis@125.236.230.94:2040') or path.startswith('sftp://dennis@125.236.230.94:2040'):
        path = path[33:]
    return path


def load_model(weights_path):
    directory_path, _ = os.path.split(weights_path)
    if os.path.isdir(weights_path):
        # need compile=False to avoid error for custom object without serialize/deserialize
        model = tf.keras.models.load_model(weights_path, custom_objects={'f1score': F1ScoreMetric}, compile=False)
        model.compile()
        print(f'Loaded and compiled model from {weights_path}')
    else:
        model_path = f'{directory_path}/model.json'
        with open(model_path, 'r') as f:
            model_config = f.read()
        model = tf.keras.models.model_from_json(model_config, custom_objects={'ModelWithCertainty': ModelWithCertainty})
        model.compile()
        model.load_weights(weights_path)
    return model
