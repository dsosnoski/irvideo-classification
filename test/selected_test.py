
import os
import sys

import numpy as np
import tensorflow as tf

from model.training_utils import load_raw_tracks, tracks_by_tag
from test.test_utils import load_tracks
from test.model_test import ModelTest


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print(tf.__version__)
    argv = sys.argv
    data_directory = argv[1]
    tracks = load_raw_tracks(f'{data_directory}/test_infos.pk')
    weights_path = argv[2]
    tag_tracks = tracks_by_tag(tracks)
    test_tracks = []
    for tag in tag_tracks.keys():
        test_tracks = test_tracks + tag_tracks[tag]
    test_tracks = [t for t in test_tracks if t.track_key in argv[3:]]
    load_tracks(test_tracks, f'{data_directory}/test_frames.npy')
    print(f'loaded {len(test_tracks)} tracks with {np.sum([t.frame_count for t in test_tracks])} frames')
    tracks_test = ModelTest(test_tracks)
    tracks_test.test(weights_path)


if __name__ == '__main__':
    sys.exit(main())
