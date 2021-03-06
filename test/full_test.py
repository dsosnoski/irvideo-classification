
import os
import sys

import dateutil as du
import numpy as np
import pandas as pd
import pickle

from model.training_utils import load_raw_tracks, tracks_by_tag
from support.data_model import CLASSES
from test.test_utils import load_tracks, convert_remote_path, flatten_tag_tracks
from test.model_test import ModelTest


def frame_count(tracks):
    return int(np.sum([t.frame_count for t in tracks]))


def all_frame_counts(tag_tracks):
    return int(np.sum([frame_count(tag_tracks[t]) for t in CLASSES]))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    argv = sys.argv
    infos_path = argv[1]
    arg_offset = 2
    save_predicts = False
    start_date = None
    while arg_offset < len(argv):
        if argv[arg_offset] == '--save_predicts':
            save_predicts = True
            arg_offset += 1
        elif argv[arg_offset] == '--start_date':
            start_date = du.parser.parse(argv[arg_offset+1])
            arg_offset += 2
        else:
            break
    test_name = os.path.split(infos_path)[1].split('_')[0]
    tracks = load_raw_tracks(infos_path)
    if start_date is not None:
        tracks = [t for t in tracks if t.start_time >= start_date]
        test_name += '_after_' + start_date.isoformat(timespec='minutes')
    tag_tracks = tracks_by_tag(tracks)

    text = f'\nTesting with {all_frame_counts(tag_tracks)} frames'
    if start_date is not None:
        text += ' from ' + start_date.isoformat()
    print(text + '\n')
    for key in CLASSES:
        print(f'{key:12}  {frame_count(tag_tracks[key]):>7}')
    print()

    test_tracks = flatten_tag_tracks(tag_tracks)
    frames_path = infos_path.replace('infos.pk', 'frames.npy')
    load_tracks(test_tracks, frames_path)
    print(f'loaded {len(test_tracks)} tracks with {np.sum([t.frame_count for t in test_tracks])} frames')
    for weights_path in argv[arg_offset:]:
        weights_path = convert_remote_path(weights_path)
        model_test = ModelTest(weights_path, test_name)
        track_predicts, frame_predicts = model_test.test(test_tracks, True)
        if save_predicts:
            frame_data = {
                'clip_id': [t.clip_key for t in test_tracks],
                'track_id': [t.track_key for t in test_tracks],
                'device_name': [t.device_name for t in test_tracks],
                'time': [t.start_time for t in test_tracks],
                'tag': [t.tag for t in test_tracks]
            }
            for i in range(len(track_predicts)):
                frame_data[f'predict {i}'] = [CLASSES[p] for p in track_predicts[i]]
            df = pd.DataFrame(frame_data)
            df.to_csv(f'{weights_path}-{test_name}-predicts.csv', index=False)
            assert len(test_tracks) == len(frame_predicts)
            for track, predicts in zip(test_tracks, frame_predicts):
                track.data = predicts
            with open(f'{weights_path}-{test_name}-predicts.pk', 'wb') as f:
                pickle.dump(test_tracks, f)


if __name__ == '__main__':
    sys.exit(main())
