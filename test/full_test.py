
import os
import sys

import numpy as np

from model.training_utils import load_raw_tracks, tracks_by_tag
from support.data_model import CLASSES, TAG_CLASS_MAP
from test.test_utils import load_tracks, convert_remote_path, flatten_tag_tracks
from test.model_test import ModelTest


def frame_count(tracks):
    return np.sum([t.frame_count for t in tracks])


def all_frame_counts(tag_tracks):
    return np.sum([frame_count(tag_tracks[t]) for t in CLASSES])


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    argv = sys.argv
    infos_path = argv[1]
    tracks = load_raw_tracks(infos_path)
    tag_tracks = tracks_by_tag(tracks)

    print(f'\nTesting with {all_frame_counts(tag_tracks)} frames:\n')
    for key in CLASSES:
        print(f'{key:12}  {frame_count(tag_tracks[key]):>7}')
    print()

    test_tracks = flatten_tag_tracks(tag_tracks)
    frames_path = infos_path.replace('infos.pk', 'frames.npy')
    load_tracks(test_tracks, frames_path)
    print(f'loaded {len(test_tracks)} tracks with {np.sum([t.frame_count for t in test_tracks])} frames')
    save_predicts = argv[2] == '--save-predicts'
    arg_offset = 3 if save_predicts else 2
    for weights_path in argv[arg_offset:]:
        weights_path = convert_remote_path(weights_path)
        model_test = ModelTest(weights_path)
        track_predicts, track_certainties = model_test.test(test_tracks, True)
        if save_predicts:
            with open(f'{weights_path}-predicts.npy', 'wb') as f:
                np.save(f, track_predicts)
            if track_certainties is not None:
                with open(f'{weights_path}-certainties.npy', 'wb') as f:
                    np.save(f, track_certainties)


if __name__ == '__main__':
    sys.exit(main())
