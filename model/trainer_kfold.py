
import datetime
import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from model.frame_sequence import FrameSequence
from model.model_builder import ModelBuilder
from model.training_utils import load_raw_tracks, tracks_by_tag, first_time_model, build_callback, \
    print_track_information, draw_figures
from support.data_model import CLASSES


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


def kfold_split(data_directory, fold_count):
    tracks = load_raw_tracks(f'{data_directory}/train_infos.pk')
    tag_tracks = tracks_by_tag(tracks)
    for tag in tag_tracks:
        print(f'{tag} loaded {len(tag_tracks[tag])} tracks')
    fold_tag_tracks = [{} for _ in range(fold_count)]
    for tag in tag_tracks.keys():
        if tag in CLASSES:
            tracks = tag_tracks[tag]
            np.random.shuffle(tracks)
            fold_tracks = [[] for _ in range(fold_count)]
            frame_counts = np.zeros(fold_count)
            for track_info in tracks:
                fold = frame_counts.argmin()
                fold_tracks[fold].append(track_info)
                frame_counts[fold] += track_info.frame_count
            for i in range(fold_count):
                fold_tag_tracks[i][tag] = fold_tracks[i]
    return fold_tag_tracks


def merge_tag_tracks(tag_tracks_list):
    merged_tracks = { c: [] for c in CLASSES }
    for tag_tracks in tag_tracks_list:
        for tag in tag_tracks:
            merged_tracks[tag] += tag_tracks[tag]
    return merged_tracks


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
    model_name_suffix = training_config.get('model_name_suffix', '')
    date_string = datetime.datetime.now().strftime('%Y%m%d')
    save_directory = f'{base_save_path}/{name}{model_name_suffix}-{date_string}'
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    print(f'saving to {save_directory}')
    epochs_count = training_config['epochs_count']

    data_directory = training_config['data_path']
    fold_count = training_config['fold_count']
    split_path = f'{save_directory}/splits.pk'
    if os.path.exists(save_directory):
        with open(split_path, 'rb') as f:
            kfold_tag_tracks = []
            for _ in range(fold_count):
                kfold_tag_tracks.append(pickle.load(f))
    else:
        kfold_tag_tracks = kfold_split(data_directory, fold_count)
        with open(split_path, 'wb') as f:
            for fold in kfold_tag_tracks:
                pickle.dump(fold, f)
    data_path = f'{data_directory}/train_frames.npy'
    batch_size = training_config['batch_size']
    max_frame_load_count = training_config['max_frame_load_count']
    frame_counts = { k: max_frame_load_count for k in CLASSES }
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
    model = ModelBuilder.build_model(input_dims, len(CLASSES), None, **model_config)
    first_time_model(model, save_directory, training_config, model_config)
    model_json = model.to_json()
    with open(f'{save_directory}/model.json', 'w') as f:
        f.write(model_json)
    for fold_num in range(fold_count):
        pass_directory = f'{save_directory}/fold{fold_num}'
        if not os.path.exists(pass_directory):
            os.mkdir(pass_directory)
            print(f'\nTraining fold {fold_num}')
            training_tracks = merge_tag_tracks([kfold_tag_tracks[i] for i in range(fold_count) if i != fold_num])
            validation_tracks = kfold_tag_tracks[fold_num]
            print_track_information(training_tracks, validation_tracks)
            train_sequence = FrameSequence.build_training_sequence(training_tracks, data_path, batch_size, tag_hots, input_dims,
                                                                   frame_counts, replace_fraction_per_epoch, noise_scale,
                                                                   rotation_limit, use_flip)
            validate_sequence = FrameSequence.build_validation_sequence(validation_tracks, data_path, batch_size, tag_hots, input_dims)
            certainty_loss_weight = training_config.get('certainty_loss_weight')
            model = ModelBuilder.build_model(input_dims, len(CLASSES), certainty_loss_weight, **model_config)
            callbacks = [build_callback(cfg, pass_directory) for cfg in training_config['callbacks']]
            history = model.fit(train_sequence, validation_data=validate_sequence, epochs=epochs_count, callbacks=callbacks)
            model.save(f'{pass_directory}/model-finished.sav')
            draw_figures(history, training_config['plots'], pass_directory)
            train_sequence.clear()
            validate_sequence.clear()
            tf.keras.backend.clear_session()


if __name__ == '__main__':
    sys.exit(main())
