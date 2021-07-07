
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import tensorflow as tf

from model.f1_metric import F1ScoreMetric
from model.frame_sequence import FrameSequence
from model.model_builder import ModelBuilder
from model.training_utils import split_training_validation, load_raw_tracks, tracks_by_tag,  print_tag_track_info
from support.data_model import CLASSES, TAG_CLASS_MAP


def first_time_model(model, save_directory, training_tracks, validation_tracks, training_config, model_config):
    print(model.summary())
    with open(f'{save_directory}/model.txt', 'w') as f:

        def summary_print(s):
            print(s, file=f)

        def print_and_write(s, f):
            print(s)
            f.write(s + '\n')

        def frame_count(tracks):
            return np.sum([t.frame_count for t in tracks])

        def all_frame_counts(tag_tracks):
            return np.sum([frame_count(tag_tracks[t]) for t in CLASSES])

        f.write('Training configuration:\n')
        f.write(json.dumps(training_config, ensure_ascii=False))
        f.write('\nModel configuration:\n')
        f.write(json.dumps(model_config, ensure_ascii=False))
        f.write('\n')
        print(model.summary(print_fn=summary_print))
        print(f'Classifier layer {model.layers[-1]} with config {json.dumps(model.layers[-1].get_config())}')
        details = f'\nTraining with {all_frame_counts(training_tracks)} frames, validating with {all_frame_counts(validation_tracks)} frames:\n'
        print(details)
        f.write(details + '\n')
        print_and_write('               Train  Validate', f)
        for key in CLASSES:
            print_and_write(f'{key:12}  {frame_count(training_tracks[key]):>7} {frame_count(validation_tracks[key]):>7}', f)


def compute_scores(tp, fp, fn):
    if tp != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = 2. * precision * recall / (precision + recall)
        return precision, recall, fscore
    else:
        return 0.0, 0.0, 0.0


def build_callback(config, save_directory):
    callback_name = config['name']
    del config['name']
    if callback_name == 'checkpoint_callback':
        checkpoint_filename = config['filepath']
        config['filepath'] = save_directory + '/' + checkpoint_filename
        print(f'saving checkpoints to {config["filepath"]}')
        return tf.keras.callbacks.ModelCheckpoint(**config)
    elif callback_name == 'lr_callback':
        return tf.keras.callbacks.ReduceLROnPlateau(**config)
    elif callback_name == 'stopping_callback':
        return tf.keras.callbacks.EarlyStopping(**config)
    else:
        raise Exception(f'Unknown callback type {callback_name}')


def main():
    argv = sys.argv
    with open(argv[1]) as f:
        training_config = json.load(f)
    with open(argv[2]) as f:
        model_config = json.load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = argv[3] if len(argv) > 3 else ''

    #check GPU and limit GPU memory growth
    gpu = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu))
    if len(gpu) > 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)

    base_save_path = training_config['base_save_path']
    name = model_config['model_name']
    date_string = datetime.datetime.now().strftime('%Y%m%d')
    save_directory = f'{base_save_path}/{name}-{date_string}'
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    print(f'saving to {save_directory}')
    epochs_count = training_config['epochs_count']

    data_directory = training_config['data_path']
    track_split_path = f'{data_directory}/track_split.pk'
    if os.path.isfile(track_split_path):
        with open(track_split_path, 'rb') as f:
            training_tracks = pickle.load(f)
            validation_tracks = pickle.load(f)
    else:
        tracks = load_raw_tracks(f'{data_directory}/train_infos.pk')
        tag_tracks = tracks_by_tag(tracks)
        print_tag_track_info(tag_tracks)
        validate_frame_counts = training_config.get('validate_frame_counts')
        if validate_frame_counts is None:
            validate_fraction = training_config['validate_fraction']
            validate_frame_counts = { k: np.sum([int(t.frame_count*validate_fraction)]) for k in validate_frame_counts for t in tag_tracks[k]}
            print(f'Computed validate frame counts {json.dumps(validate_frame_counts)}')
        training_tracks, validation_tracks = split_training_validation(tag_tracks, validate_frame_counts)
        save_split_path = f'{save_directory}/track_split.pk'
        with open(save_split_path, 'wb') as f:
            pickle.dump(training_tracks, f)
            pickle.dump(validation_tracks, f)

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
    tag_hots = {}
    tag_indexes = {}
    for index, tag in enumerate(CLASSES):
        one_hot = np.zeros(len(CLASSES))
        one_hot[index] = 1
        tag_hots[tag] = one_hot
        tag_indexes[tag] = index
    train_sequence = FrameSequence.build_training_sequence(training_tracks, data_path, batch_size, tag_hots, input_dims,
                                                           frame_counts, replace_fraction_per_epoch, noise_scale,
                                                           rotation_limit, use_flip)
    validate_sequence = FrameSequence.build_validation_sequence(validation_tracks, data_path, batch_size, tag_hots, input_dims)

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
        first_time_model(model, save_directory, training_tracks, validation_tracks, training_config, model_config)
        model_json = model.to_json()
        with open(f'{save_directory}/model.json', 'w') as f:
            f.write(model_json)
        print(f'created new model')

    try:
        callbacks = [build_callback(cfg, save_directory) for cfg in training_config['callbacks']]
        history = model.fit(train_sequence, validation_data=validate_sequence, epochs=epochs_count, callbacks=callbacks)
        model.save(f'{save_directory}/model-finished.sav')

        plots = training_config['plots']
        plt.figure(figsize=(8, 6*len(plots)))
        plt_position = len(plots) * 100 + 11
        for i, plot in enumerate(plots):
            plt.subplot(plt_position + i)
            plt.title(plot['title'])
            legends = []
            for value in plot['values']:
                plt.plot(history.history[value])
                legend = value.replace('_', ' ').title()
                legends.append('Training ' + legend)
                value = 'val_' + value
                plt.plot(history.history[value])
                legends.append('Validation ' + legend)
            plt.ylabel(plot['y-label'])
            plt.xlabel('Epoch')
            plt.legend(legends, loc=plot['caption-loc'], framealpha=.5)
        plt.savefig(f'{save_directory}/history.png')
        plt.close()

        tf.keras.backend.clear_session()
        return 0

    except KeyboardInterrupt:
        model.save(f'{save_directory}/model-interrupt.sav')
        print('Shutdown requested...exiting with model saved')
        sys.exit(0)


if __name__ == '__main__':
    sys.exit(main())
