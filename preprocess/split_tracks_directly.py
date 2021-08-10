import os
import pickle
import sys
import traceback

import h5py
import numpy as np
from dateutil.parser import parse as parse_date

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from preprocess.image_sizing import IMAGE_DIMENSION
from support.data_model import Track
from support.track_utils import convert_frames, extract_hdf5_frames, extract_hdf5_crops, prune_frames, smooth_resizer

# DATASET_PATH = '/data/cacophony/ai-dataset/dataset.hdf5'
# OUTPUT_DIRECTORY = '/data/dennis/irvideo/new-data'
# DATASET_PATH = '/mnt/old/irvideos/dataset.hdf5'
# OUTPUT_DIRECTORY = '/mnt/old/irvideos'

DATA_FILE_NAMES = ['test_frames.npy', 'train_frames.npy']
METADATA_NAMES = ['test_infos.pk', 'train_infos.pk']


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
    frame_count = 0
    test_count = 0
    train_count = 0
    discard_count = 0
    ftest_data, ftrain_data = open_files(output_directory, DATA_FILE_NAMES)
    ftest_meta, ftrain_meta = open_files(output_directory, METADATA_NAMES)
    ratiofn = smooth_resizer(IMAGE_DIMENSION)
    for clip_key in clips_hdf5:
        clip_hdf5 = clips_hdf5[clip_key]
        start_time = parse_date(clip_hdf5.attrs['start_time'])
        device_name = clip_hdf5.attrs.get('device')
        location = clip_hdf5.attrs.get('location')
        tkeys = [k for k in clip_hdf5 if not k == 'background_frame']
        bgframe = np.array(clip_hdf5['background_frame'])
        for track_key in tkeys:
            try:
                track_hdf5 = clip_hdf5[track_key]
                hdf5_bounds = track_hdf5.attrs['bounds_history']
                tag = track_hdf5.attrs['tag']
                frames = extract_hdf5_frames(track_hdf5['original'])
                crops = extract_hdf5_crops(track_hdf5['cropped'])
                masses = track_hdf5.attrs['mass_history']
                keepfn = lambda a, b: b > (2 + a*.005)
                _, frames, bounds, masses, pixels = prune_frames(crops, frames, hdf5_bounds, masses, clip_key, track_key, keepfn)
                if len(bounds) > 0:
                    aframes, obounds, ratios = convert_frames(frames, bgframe, bounds, IMAGE_DIMENSION, ratiofn)
                    if clip_key in test_clips:
                        test_clips[clip_key] = True
                        offset = ftest_data.tell()
                        track = Track(tag, clip_key, track_key, start_time, device_name, location, masses, pixels, bounds, obounds, ratios, offset)
                        np.save(ftest_data, aframes)
                        test_count += 1
                        pickle.dump(track, ftest_meta)
                    else:
                        offset = ftrain_data.tell()
                        track = Track(tag, clip_key, track_key, start_time, device_name, location, masses, pixels, bounds, obounds, ratios, offset)
                        np.save(ftrain_data, aframes)
                        train_count += 1
                        pickle.dump(track, ftrain_meta)
                    track_count += 1
                    frame_count += len(obounds)
                    if track_count % 1000 == 0:
                        print(f'processed {track_count} tracks with {frame_count} frames, {train_count} training tracks and  {test_count} test tracks; {discard_count} tracks discarded')
                        ftest_data, ftrain_data = save_and_reopen_files([ftest_data, ftrain_data])
                else:
                    discard_count += 1
                    print(f'discarding zero-length {clip_key}-{track_key} in test set {clip_key in test_clips}')

            except KeyboardInterrupt:
                print('Shutdown requested by keyboard interrupt, terminating...')
                sys.exit(0)
            except Exception as e:
                print(f'Exception processing {clip_key}-{track_key}: {e}')
                traceback.print_exc()

    close_files([ftest_data, ftrain_data])
    close_files([ftest_meta, ftrain_meta])
    print(f'Completed processing of {track_count} tracks with {frame_count} frames, {train_count} training tracks and {test_count} test tracks; {discard_count} tracks discarded')
    missing_clips = [k for k,v in test_clips.items() if not v]
    for clip_key in missing_clips:
        print(f'missing specified test clip {clip_key}')


if __name__ == '__main__':
    sys.exit(main())
