import os
import traceback

import h5py
import numpy as np
import pickle
import sys
from dateutil.parser import parse as parse_date

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from preprocess.image_sizing import IMAGE_DIMENSION, IMAGE_RESIZE_RATIOS
from support.data_model import Track
from support.track_utils import convert_frames, extract_hdf5_frames, extract_hdf5_crops, prune_frames, stepped_resizer

# DATASET_PATH = '/data/cacophony/ai-dataset/dataset.hdf5'
# OUTPUT_DIRECTORY = '/data/dennis/irvideo/new-data'
# DATASET_PATH = '/mnt/old/irvideos/dataset.hdf5'
# OUTPUT_DIRECTORY = '/mnt/old/irvideos'


def open_files(root_path, names):
    paths = [f'{root_path}/{n}' for n in names]
    for path in paths:
        if os.path.isfile(path):
            os.remove(path)
    files = [open(p, 'a+b') for p in paths]
    return files


def close_files(files):
    for file in files:
        file.close()


def save_and_reopen_files(files):
    paths = [f.name for f in files]
    close_files(files)
    files = [open(p, 'a+b') for p in paths]
    return files


def main():
    argv = sys.argv
    dataset_path = argv[1]
    output_directory = argv[2]

    file_hdf5 = h5py.File(dataset_path, 'r')
    clips_hdf5 = file_hdf5['clips']

    with open(f'{output_directory}/testset.txt', 'r') as f:
        lines = f.readlines()
    test_clips = {}
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            if len(line) > 6:
                print(f'invalid segment id {line} (position {len(test_clips)})')
            else:
                test_clips[line] = False

    track_count = 0
    test_count = 0
    train_count = 0
    newer_count = 0
    cutoff_time = parse_date('2021-03-29T08:07:54.240643+13:00')
    for clip_key in clips_hdf5:
        clip_hdf5 = clips_hdf5[clip_key]
        start_time = parse_date(clip_hdf5.attrs['start_time'])
        tkeys = [k for k in clip_hdf5 if not k == 'background_frame']
        if clip_key in test_clips:
            test_clips[clip_key] = True
            test_count += len(tkeys)
        elif start_time < cutoff_time:
            train_count += len(tkeys)
        else:
            newer_count += len(tkeys)
        track_count += len(tkeys)
    print(f'Found {track_count} tracks total: {train_count} training tracks, {test_count} test tracks, and {newer_count} newer tracks')
    missing_clips = [k for k,v in test_clips.items() if not v]
    for clip_key in missing_clips:
        print(f'missing specified test clip {clip_key}')


if __name__ == '__main__':
    sys.exit(main())
