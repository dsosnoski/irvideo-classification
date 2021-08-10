
import datetime
import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from model.f1_metric import F1ScoreMetric
from model.training_utils import flatten_tag_tracks, first_time_model, build_callback, draw_figures, dense_norm_relu
from rnnmodel.masked_frame_sequence import RnnFrameSequence
from rnnmodel.model_builder import build_model
from support.data_model import CLASSES


def _center_and_size(bound):
    l, t, r, b = bound
    return np.array([t+b/2, l+r/2, (b-t+r-l)/2])


def _dims(bound):
    l, t, r, b = bound
    return b-t, r-l


def build_sample_data(track, predicts, value_selects, **scale_params):
    """
    Build frame sample data for track, as one or more inputs. The inputs consist of any of:
    1) Predict values (always [0,1])
    2) Normalized pixel count [0,1] (actual count is divided by pixel_count_scale and clipped at 1), base and squared
    3) Width, height, average, size [0,1] where actual dimension is divided by dimension_scale and clipped at 1
    4) Horizontal movement, vertical movement, average width/height difference [-1,1] where actual change in center
    position/size is divided by average width/height and multiplied by movement_scale, then clipped to [-1,1]
    :param track: track information
    :param predicts: classification prediction values for frames
    :param value_selects: arrays of values included in samples, any of 'predicted', 'pixel_count', 'dimensions',
     'movement'; if more than one array, sample data is returned as a list of arrays
    :param scale_params: scaling parameters
    :return sample data (one row per frame)
    """
    sample_data = [[] for _ in value_selects]
    last_pos = np.array(_center_and_size(track.crop_bounds[0]))
    for frame_index in range(0, track.frame_count):
        new_pos = _center_and_size(track.crop_bounds[frame_index])
        avg_dim = new_pos[2]
        delta_pos = new_pos - last_pos
        last_pos = new_pos
        for select_index, labels in enumerate(value_selects):
            selected_data = []
            for label in labels:
                if label == 'predicted':
                    selected_data += list(predicts[frame_index])
                elif label == 'pixel_count':
                    pixel_value = min(track.pixels[frame_index] / scale_params['pixel_count_scale'], 1)
                    selected_data.append(pixel_value)
                    selected_data.append(pixel_value**2)
                elif label == 'dimensions':
                    h, w = _dims(track.crop_bounds[frame_index])
                    h_scaled = h / scale_params['dimension_scale']
                    w_scaled = w / scale_params['dimension_scale']
                    selected_data.append(min(h_scaled, 1))
                    selected_data.append(min(w_scaled, 1))
                    selected_data.append(min(avg_dim / scale_params['dimension_scale'], 1))
                    selected_data.append(min(h_scaled * w_scaled, 1))
                elif label == 'movement':
                    selected_data += list(np.clip(delta_pos * scale_params['movement_scale'] / avg_dim, -1, 1))
                else:
                    raise ValueError(f'Unknown value selection code "{label}"')
            sample_data[select_index].append(selected_data)
    return [np.array(d) for d in sample_data]


def main():
    argv = sys.argv
    with open(argv[1]) as f:
        training_config_text = f.read()
    training_config = json.loads(training_config_text)
    with open(argv[2]) as f:
        model_config_text = f.read()
    model_config = json.loads(model_config_text)
    predicts_path = argv[3]
    track_split_path = argv[4]
    os.environ['CUDA_VISIBLE_DEVICES'] = argv[5] if len(argv) > 5 else ''

    #check GPU and limit GPU memory growth
    gpu = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu))
    if len(gpu) > 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)

    base_save_path = training_config['base_save_path']
    name = model_config['model_name']
    model_name_suffix = training_config.get('model_name_suffix', '')
    date_string = datetime.datetime.now().strftime('%Y%m%d')
    save_directory = f'{base_save_path}/{name}{model_name_suffix}-{date_string}'
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    print(f'saving to {save_directory}')
    epochs_count = training_config['epochs_count']

    with open(predicts_path, 'rb') as f:
        full_data = pickle.load(f)
    track_data = {(t.clip_key, t.track_key): t.data for t in full_data}
    with open(track_split_path, 'rb') as f:
        training_tracks = flatten_tag_tracks(pickle.load(f))
        validation_tracks = flatten_tag_tracks(pickle.load(f))
    input_values = training_config['input_values']
    input_scales = training_config['input_scales']
    for track in training_tracks:
        track.data = build_sample_data(track, track_data[(track.clip_key, track.track_key)], input_values, **input_scales)
    for track in validation_tracks:
        track.data = build_sample_data(track, track_data[(track.clip_key, track.track_key)], input_values, **input_scales)
    max_frame_count = training_config['max_frame_count']
    input_dims = []
    for input_value in training_tracks[0].data:
        input_dims.append((max_frame_count, input_value.shape[1]))
    batch_size = training_config['batch_size']
    noise_scales = training_config.get('noise_scales')
    per_frame_inputs = [v != ['time_of_data'] for v in input_values]
    train_sequence = RnnFrameSequence(training_tracks, batch_size, max_frame_count, per_frame_inputs, noise_scales)
    validate_sequence = RnnFrameSequence(validation_tracks, batch_size, max_frame_count, per_frame_inputs)
    model = build_model(input_dims, len(CLASSES), custom_objects={'F1ScoreMetric': F1ScoreMetric}, **model_config)
    first_time_model(model, training_config_text, model_config_text, save_directory)
    model_json = model.to_json()
    with open(f'{save_directory}/model.json', 'w') as f:
        f.write(model_json)
    try:
        callbacks = [build_callback(cfg, save_directory) for cfg in training_config['callbacks']]
        history = model.fit(train_sequence, validation_data=validate_sequence, epochs=epochs_count, callbacks=callbacks)
        model.save(f'{save_directory}/model-finished.sav')
        draw_figures(history, training_config['plots'], save_directory)
        tf.keras.backend.clear_session()
        return 0

    except KeyboardInterrupt:
        model.save(f'{save_directory}/model-interrupt.sav')
        print('Shutdown requested...exiting with model saved')
        sys.exit(0)


if __name__ == '__main__':
    sys.exit(main())
