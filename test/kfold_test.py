
import os
import pickle
import sys
import traceback

import numpy as np

from support.data_model import CLASSES, TAG_CLASS_MAP
from test.test_utils import load_tracks, load_model, prep_track, flatten_tag_tracks


def frame_count(tracks):
    return np.sum([t.frame_count for t in tracks])


def all_frame_counts(tag_tracks):
    return np.sum([frame_count(tag_tracks[t]) for t in CLASSES])


def load_tracks(tracks, data_path):
    with open(data_path, 'rb') as f:
        for track in tracks:
            f.seek(track.offset)
            track.data = np.load(f)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    argv = sys.argv
    data_path = argv[1]
    fold_root = argv[2]
    with open(f'{fold_root}/splits.pk', 'rb') as f:
        kfold_tag_tracks = []
        try:
            while True:
                kfold_tag_tracks.append(pickle.load(f))
        except Exception as e:
            traceback.print_exc()
            pass
    fold_count = len(kfold_tag_tracks)
    class_to_index = {c: i for i, c in enumerate(CLASSES)}
    track_predicts = []
    track_infos = []
    frames_correct = 0
    frames_wrong = 0
    for fold_number in range(fold_count):
        test_tracks = flatten_tag_tracks(kfold_tag_tracks[fold_number])
        track_infos += test_tracks
        load_tracks(test_tracks, f'{data_path}/train_frames.npy')
        fold_directory = f'{fold_root}/fold{fold_number}'
        saved_models = [n for n in os.listdir(fold_directory) if n.startswith('model-') and not n.endswith('.sav') and not n.endswith('.png')]
        weights_name = sorted(saved_models, key=lambda n: int(n.split('-')[1]))[-1]
        print(f'fold {fold_number} using model {weights_name}')
        model = load_model(f'{fold_directory}/{weights_name}')
        sample_dims = tuple([d for d in model.layers[0].output.shape[1:]])
        fold_correct = 0
        fold_wrong = 0
        for track in test_tracks:
            tag = TAG_CLASS_MAP[track.tag]
            samples = prep_track(track, sample_dims)
            predicts = model.predict(samples)
            track_predicts.append(predicts)
            frame_maxes = np.argmax(predicts, axis=1)
            corrects = frame_maxes == class_to_index[tag]
            num_correct = np.sum(corrects)
            num_wrong = len(frame_maxes) - num_correct
            fold_correct += num_correct
            fold_wrong += num_wrong
        total_frames = fold_correct + fold_wrong
        print(f'frames {fold_correct} predicted correctly ({fold_correct/total_frames:.4f}), {fold_wrong} predicted incorrectly ({fold_wrong/total_frames:.4f})')
        frames_correct += fold_correct
        frames_wrong += fold_wrong
    print(f'Across all folds {fold_correct} predicted correctly ({fold_correct/total_frames:.4f}), {fold_wrong} predicted incorrectly ({fold_wrong/total_frames:.4f})')
    for track in track_infos:
        track.data = None
    with open(f'{fold_root}/fold_results.pk', 'wb') as f:
        pickle.dump(track_infos, f)
        pickle.dump(track_predicts, f)


if __name__ == '__main__':
    sys.exit(main())
