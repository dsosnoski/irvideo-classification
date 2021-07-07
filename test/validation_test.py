
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sn
import sys
import tensorflow as tf

from model.f1_metric import F1ScoreMetric
from support.data_model import CLASSES, TAG_CLASS_MAP
from test.test_utils import prep_track, load_tracks


def evaluate(predicts, actuals, base_path, title):
    confusion_matrix = np.zeros((len(CLASSES), len(CLASSES)))
    for predict, actual in zip(predicts, actuals):
        confusion_matrix[actual, predict] += 1
    normalized_matrix = (confusion_matrix.T / confusion_matrix.sum(axis=1)).T
    precision_row = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)
    display_matrix = np.row_stack((normalized_matrix, precision_row))
    num_correct = np.diag(confusion_matrix).sum()
    num_total = len(predicts)
    num_wrong = num_total - num_correct
    print(f'{title} gives total of {num_correct} tracks predicted correctly ({num_correct/num_total:.4f}), {num_wrong} predicted incorrectly ({num_wrong/num_total:.4f})')
    print(f' mean accuracy (recall) {np.diag(normalized_matrix).sum()/len(CLASSES):.4f}, mean precision {np.sum(precision_row)/len(CLASSES):.4f}')
    print(confusion_matrix)
    plt.figure(figsize=(10,10))
    sn.heatmap(display_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES + ['precision'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.suptitle(f'overall accuracy: {num_correct/num_total:.4f}, mean recall: {np.diag(normalized_matrix).sum()/len(CLASSES):.4f}, mean precision: {np.sum(precision_row)/len(CLASSES):.4f}')
    plt.savefig(f'{base_path}-{title.replace(" ", "_")}validation-results.png')
    plt.close()


def sum_weighted(predicts, weights):
    return np.matmul(weights.T, predicts)


def test(tracks, weights_path):
    directory_path, _ = os.path.split(weights_path)
    model_path = f'{directory_path}/model-finished.sav'
    if os.path.isdir(model_path):
        # need compile=False to avoid error for custom object without serialize/deserialize
        model = tf.keras.models.load_model(model_path, custom_objects={'f1score': F1ScoreMetric}, compile=False)
        model.compile()
        model.load_weights(weights_path)
    elif os.path.isdir(weights_path):
        # need compile=False to avoid error for custom object without serialize/deserialize
        model = tf.keras.models.load_model(weights_path, custom_objects={'f1score': F1ScoreMetric}, compile=False)
        model.compile()
    else:
        model_path = f'{directory_path}/model.json'
        with open(model_path, 'r') as f:
            model_config = f.read()
        model = tf.keras.models.model_from_json(model_config)
        model.compile()
        model.load_weights(weights_path)
    sample_dims = tuple([d for d in model.layers[0].output.shape[1:]])
    print(f'Testing model {weights_path} with input sample dimensions {sample_dims}:')
    class_to_index = { c:i for i,c in enumerate(CLASSES) }
    frames_correct = 0
    frames_wrong = 0
    actuals = []
    frame_mean_predicts = []
    frame_squared_predicts = []
    frame_pixelcount_predicts = []
    frame_pixelcount_sqrpredicts = []
    for track in tracks:
        if track.frame_count > 0:
            tag = TAG_CLASS_MAP[track.tag]
            actual = class_to_index[tag]
            actuals.append(actual)
            samples = prep_track(track, sample_dims)
            predicts = model.predict(samples)
            print(f'\nTrack {track.track_key} tagged {track.tag} with {len(predicts)} frames:')
            print('Frame     ' + ('{:<12}  '*len(CLASSES)).format(*CLASSES))
            flag_chars = { (True, True): '*', (True, False): '-', (False, True): '>', (False, False): ' '}
            for i in range(len(predicts)):
                highest = np.argmax(predicts[i])
                values = [(p, flag_chars[j == actual, j == highest]) for j, p in enumerate(predicts[i])]
                values = [v for t in values for v in t]
                print(f'{i:>5}:    ' + ('{:<3.2f}{}         '*len(CLASSES)).format(*values) + ('right' if highest == actual else 'wrong'))
            frame_mean_predicts.append(np.argmax(predicts.sum(axis=0)))
            predicts_squared = predicts**2
            frame_squared_predicts.append(np.argmax(predicts_squared.sum(axis=0)))
            pixelcount_weights = np.array([(f > 0).sum() for f in samples])
            frame_pixelcount_predicts.append(np.argmax(sum_weighted(predicts, pixelcount_weights)))
            frame_pixelcount_sqrpredicts.append(np.argmax(sum_weighted(predicts_squared, pixelcount_weights)))
            frame_maxes = np.argmax(predicts, axis=1)
            corrects = frame_maxes == class_to_index[tag]
            num_correct = np.sum(corrects)
            num_wrong = len(frame_maxes) - num_correct
            frames_correct += num_correct
            frames_wrong += num_wrong
            print(f'Frame predictions {num_correct/len(frame_maxes):.4f} correct and {num_wrong/len(frame_maxes):.3f} wrong ({num_correct} vs {num_wrong})')
            matches = np.array([frame_mean_predicts[-1], frame_squared_predicts[-1], frame_pixelcount_sqrpredicts[-1], frame_pixelcount_sqrpredicts[-1]]) == actual
            print(f'Tag {tag} {"all correct" if all(matches) else "partial correct" if any(matches) else "all wrong"}: Mean predict {CLASSES[frame_mean_predicts[-1]]}, squared mean predict {CLASSES[frame_squared_predicts[-1]]}, pixel count weighted squared mean predict {CLASSES[frame_pixelcount_sqrpredicts[-1]]}')
    total_frames = frames_correct + frames_wrong
    print(f'frames {frames_correct} predicted correctly ({frames_correct/total_frames:.4f}), {frames_wrong} predicted incorrectly ({frames_wrong/total_frames:.4f})')
    figure_path = weights_path if weights_path else model_path
    evaluate(frame_mean_predicts, actuals, figure_path, 'frame mean')
    evaluate(frame_squared_predicts, actuals, figure_path, 'frame squared predicts mean')
    evaluate(frame_pixelcount_predicts, actuals, figure_path, 'nonzero pixel count weighting')
    evaluate(frame_pixelcount_sqrpredicts, actuals, figure_path, 'nonzero pixel count weighting squared predicts')


def frame_count(tracks):
    return np.sum([t.frame_count for t in tracks])


def all_frame_counts(tag_tracks):
    return np.sum([frame_count(tag_tracks[t]) for t in CLASSES])


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    argv = sys.argv
    data_directory = argv[1]
    for weights_path in argv[2:]:
        if weights_path.endswith('.index'):
            weights_path = weights_path[:-6]
        if weights_path.startswith('fish://dennis@125.236.230.94:2040') or weights_path.startswith('sftp://dennis@125.236.230.94:2040'):
            weights_path = weights_path[33:]
        directory_path, _ = os.path.split(weights_path)
        track_split_path = f'{directory_path}/track_split.pk'
        with open(track_split_path, 'rb') as f:
            pickle.load(f)
            validation_tracks = pickle.load(f)
        test_tracks = []
        for tag in validation_tracks.keys():
            if tag in TAG_CLASS_MAP:
                tracks = validation_tracks[tag]
                test_tracks = test_tracks + tracks
            else:
                print(f'skipping {len(validation_tracks[tag])} tracks with unsupported tag {tag}')
        load_tracks(test_tracks, f'{data_directory}/train_data.npy')
        print(f'loaded {len(test_tracks)} tracks with {np.sum([len(t.frames) for t in test_tracks])} frames')
        test(test_tracks, weights_path)


if __name__ == '__main__':
    sys.exit(main())
