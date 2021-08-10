
import datetime
import dateutil.parser as dtparser
import json
import numpy as np
import os
import pickle
import sys
import tensorflow as tf

from model.f1_metric import F1ScoreMetric
from model.frame_sequence import FrameSequence
from model.model_builder import ModelBuilder
from model.training_utils import split_training_validation, load_raw_tracks, tracks_by_tag, print_tag_track_info, \
    first_time_model, build_callback, draw_figures, print_track_information
from support.data_model import CLASSES, TAG_HOTS


def get_training_validation_split(data_directory, save_directory, training_config):
    track_split_path = f'{data_directory}/track_split.pk'
    if os.path.isfile(track_split_path):
        with open(track_split_path, 'rb') as f:
            training_tracks = pickle.load(f)
            validation_tracks = pickle.load(f)
    else:
        tracks = load_raw_tracks(f'{data_directory}/train_infos.pk')
        cutoff_time = dtparser.parse(training_config['cutoff_time'])
        tracks = [t for t in tracks if t.start_time < cutoff_time]
        tag_tracks = tracks_by_tag(tracks)
        print(f'Using training data prior to {cutoff_time.isoformat(timespec="minutes")}')
        print_tag_track_info(tag_tracks)
        validate_frame_counts = training_config.get('validate_frame_counts')
        if validate_frame_counts is None:
            validate_fraction = training_config['validate_fraction']
            validate_frame_counts = {k: np.sum([int(t.frame_count * validate_fraction)]) for k in validate_frame_counts
                                     for t in tag_tracks[k]}
            print(f'Computed validate frame counts {json.dumps(validate_frame_counts)}')
        training_tracks, validation_tracks = split_training_validation(tag_tracks, validate_frame_counts)
        save_split_path = f'{save_directory}/track_split.pk'
        with open(save_split_path, 'wb') as f:
            pickle.dump(training_tracks, f)
            pickle.dump(validation_tracks, f)
    return training_tracks, validation_tracks


def main():
    argv = sys.argv
    with open(argv[1]) as f:
        training_config_text = f.read()
    training_config = json.loads(training_config_text)
    with open(argv[2]) as f:
        model_config_text = f.read()
    model_config = json.loads(model_config_text)

    os.environ['CUDA_VISIBLE_DEVICES'] = argv[3] if len(argv) > 3 else ''

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

    data_directory = training_config['data_path']
    training_tracks, validation_tracks = get_training_validation_split(data_directory, save_directory, training_config)

    data_path = f'{data_directory}/train_frames.npy'
    batch_size = training_config['batch_size']
    frame_count_multiple = training_config.get('frame_count_multiple')
    max_frame_load_count = training_config['max_frame_load_count']
    sample_weights = training_config.get('sample_weights')
    if sample_weights is None:
        frame_counts = { k: max_frame_load_count for k in CLASSES }
    else:
        frame_counts = { k: min(v * frame_count_multiple, max_frame_load_count) for k, v in sample_weights.items() }
    replace_fraction_per_epoch = training_config['replace_fraction_per_epoch']
    noise_scale = training_config.get('noise_scale')
    rotation_limit = training_config.get('rotation_limit')
    use_flip = training_config.get('use_flip', False)
    input_dims = tuple(training_config['input_dims'])
    train_sequence = FrameSequence.build_training_sequence(training_tracks, data_path, batch_size, TAG_HOTS, input_dims,
                                                           frame_counts, replace_fraction_per_epoch, noise_scale,
                                                           rotation_limit, use_flip)
    validate_sequence = FrameSequence.build_validation_sequence(validation_tracks, data_path, batch_size, TAG_HOTS, input_dims)
    if len(argv) > 4:
        model_argument = argv[4]
        if model_argument.endswith('.sav'):
            model = tf.keras.models.load_model(model_argument)
            print(f'restored existing model {model_argument}')
        else:
            directory_path, _ = os.path.split(model_argument)
            with open(f'{directory_path}/model.json', 'r') as f:
                json_config = f.read()
            model = tf.keras.models.model_from_json(json_config)
            optimizer = tf.keras.optimizers.deserialize(model_config['optimizer'])
            loss = tf.keras.losses.deserialize(model_config['loss'])
            metrics = [tf.keras.metrics.deserialize(c, custom_objects={'F1ScoreMetric': F1ScoreMetric}) for c in model_config['metrics']]
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            model.load_weights(model_argument)
            print(f'loaded existing model and weights {model_argument}')
    else:
        certainty_loss_weight = training_config.get('certainty_loss_weight')
        model = ModelBuilder.build_model(input_dims, len(CLASSES), certainty_loss_weight, **model_config)
        first_time_model(model, training_config_text, model_config_text, save_directory)
        model_json = model.to_json()
        with open(f'{save_directory}/model.json', 'w') as f:
            f.write(model_json)
        print(f'created new model')
    print_track_information(training_tracks, validation_tracks)
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
