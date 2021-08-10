import json
import os
import sys

import dateutil as du
import numpy as np
import pandas as pd
import pickle

from model.training_utils import load_raw_tracks, tracks_by_tag
from support.data_model import CLASSES, TAG_INDEXES
from test.test_utils import load_tracks, convert_remote_path, flatten_tag_tracks, load_model
from test.model_test import ModelTest


def frame_count(tracks):
    return int(np.sum([t.frame_count for t in tracks]))


def all_frame_counts(tag_tracks):
    return int(np.sum([frame_count(tag_tracks[t]) for t in CLASSES]))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    argv = sys.argv
    infos_path = argv[1]
    with open(argv[2]) as f:
        training_config = json.load(f)
    predicts_path = argv[3]
    arg_offset = 4
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
    with open(predicts_path, 'rb') as f:
        full_data = pickle.load(f)
    track_data = {(t.clip_key, t.track_key): t.data for t in full_data}
    test_tracks = load_raw_tracks(infos_path)
    if start_date is not None:
        test_tracks = [t for t in test_tracks if t.start_time >= start_date]
        test_name += '_after_' + start_date.isoformat(timespec='minutes')
    input_values = training_config['input_values']
    input_scales = training_config['input_scales']
    for track in test_tracks:
        track.data = build_sample_data(track, track_data[(track.clip_key, track.track_key)], input_values, **input_scales)
    actuals = [TAG_INDEXES[t.tag] for t in test_tracks]

    text = f'\nTesting with {len(track)} tracks'
    if start_date is not None:
        text += ' from ' + start_date.isoformat()
    print(text + '\n')
    for key in CLASSES:
        print(f'{key:12}  {frame_count(tag_tracks[key]):>7}')
    print()

    for weights_path in argv[arg_offset:]:
        weights_path = convert_remote_path(weights_path)
        model = load_model(weights_path)

        for track in test_tracks:
            predicts = model.predict(track.data)
            class_index = np.argmax(predicts)

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
