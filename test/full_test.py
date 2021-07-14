
import os
import sys

import numpy as np

from model.training_utils import load_raw_tracks, tracks_by_tag
from support.data_model import CLASSES, TAG_CLASS_MAP
from test.test_utils import load_tracks
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

    test_tracks = []
    for tag in tag_tracks.keys():
        if tag in TAG_CLASS_MAP:
            tracks = tag_tracks[tag]
            test_tracks = test_tracks + tracks
        else:
            print(f'skipping {len(tag_tracks[tag])} tracks with unsupported tag {tag}')

    frames_path = infos_path.replace('infos.pk', 'frames.npy')
    load_tracks(test_tracks, frames_path)
    print(f'loaded {len(test_tracks)} tracks with {np.sum([t.frame_count for t in test_tracks])} frames')
    for weights_path in argv[2:]:
        if weights_path.endswith('.index'):
            weights_path = weights_path[:-6]
        if weights_path.startswith('fish://dennis@125.236.230.94:2040') or weights_path.startswith('sftp://dennis@125.236.230.94:2040'):
            weights_path = weights_path[33:]
        model_test = ModelTest(weights_path)
        model_test.test(test_tracks, True)


if __name__ == '__main__':
    sys.exit(main())
