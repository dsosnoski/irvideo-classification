
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import traceback

from support.data_model import TAG_CLASS_MAP, CLASSES


def load_raw_tracks(path):
    tracks = []
    with open(path, 'rb') as f:
        try:
            while True:
                tracks.append(pickle.load(f))
        except Exception as e:
            traceback.print_exc()
            pass
    return tracks


def tracks_by_tag(tracks):
    tag_tracks = {t: [] for t in CLASSES}
    for track in tracks:
        if track.tag in TAG_CLASS_MAP:
            track.tag = TAG_CLASS_MAP[track.tag]
            tag_tracks[track.tag].append(track)
    return tag_tracks


def flatten_tag_tracks(tag_tracks):
    flat_tracks = []
    for tracks in tag_tracks.values():
        flat_tracks += tracks
    return flat_tracks


def print_tag_track_info(infos):
    for k in infos:
        tracks = infos[k]
        fcount = np.sum([t.frame_count for t in tracks])
        print(f'{k}: {len(tracks)} tracks with {fcount} frames')


def split_training_validation(tag_tracks, validate_frame_counts):
    train_tracks = {}
    validate_tracks = {}
    for tag in tag_tracks.keys():
        if tag in CLASSES:
            tracks = tag_tracks[tag]
            np.random.shuffle(tracks)
            vcount = 0
            train_use = []
            validate_use = []
            for track_info in tracks:
                if vcount < validate_frame_counts[tag]:
                    validate_use.append(track_info)
                    vcount += track_info.frame_count
                else:
                    train_use.append(track_info)
            train_tracks[tag] = train_use
            validate_tracks[tag] = validate_use
    return train_tracks, validate_tracks


def first_time_model(model, training_config_text, model_config_text, save_directory):
    print(model.summary())
    with open(f'{save_directory}/model.txt', 'w') as f:

        def summary_print(s):
            print(s, file=f)

        f.write('\nTraining configuration:\n' + training_config_text + '\n')
        f.write('\nModel configuration:\n' + model_config_text + '\n')
        print(model.summary(print_fn=summary_print))
        tf.keras.utils.plot_model(model, to_file=f'{save_directory}/model.png', show_shapes=True)


def frame_count(tracks):
    return int(np.sum([t.frame_count for t in tracks]))


def all_frame_counts(tag_tracks):
    return int(np.sum([frame_count(tag_tracks[t]) for t in CLASSES]))


def print_track_information(training_tracks, validation_tracks):
    details = f'\nTraining with {all_frame_counts(training_tracks)} frames, validating with {all_frame_counts(validation_tracks)} frames:\n'
    print(details)
    print('               Train  Validate')
    for key in CLASSES:
        print(f'{key:12}  {frame_count(training_tracks[key]):>7} {frame_count(validation_tracks[key]):>7}')


def dense_norm_relu(n, x):
    x = tf.keras.layers.Dense(n, kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)


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
    config_copy = config.copy()
    del config_copy['name']
    if callback_name == 'checkpoint_callback':
        checkpoint_filename = config_copy['filepath']
        config_copy['filepath'] = save_directory + '/' + checkpoint_filename
        print(f'saving checkpoints to {config_copy["filepath"]}')
        return tf.keras.callbacks.ModelCheckpoint(**config_copy)
    elif callback_name == 'lr_callback':
        return tf.keras.callbacks.ReduceLROnPlateau(**config_copy)
    elif callback_name == 'stopping_callback':
        return tf.keras.callbacks.EarlyStopping(**config_copy)
    else:
        raise Exception(f'Unknown callback type {callback_name}')


def draw_figures(history, plots, save_directory):
    plt.figure(figsize=(8, 6 * len(plots)))
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
        plt.xlim(left=1)
        plt.ylim(0.0,1.0)
        plt.ylabel(plot['y-label'])
        plt.xlabel('Epoch')
        plt.legend(legends, loc=plot['caption-loc'], framealpha=.5)
    plt.savefig(f'{save_directory}/history.png')
    plt.close()
