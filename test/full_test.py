
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sn
import sys
import tensorflow as tf

from model.f1_metric import F1ScoreMetric
from model.training_utils import load_raw_tracks, print_tag_track_info, tracks_by_tag
from support.data_model import CLASSES, TAG_CLASS_MAP
from test.test_utils import prep_track, load_tracks


class PredictionVariation:

    def __init__(self, description, evaluatefn):
        self.description = description
        self.evaluatefn = evaluatefn
        self.accumulated_results = []

    def evaluate(self, predicts, certainties, track):
        self.accumulated_results.append(np.argmax(self.evaluatefn(predicts, certainties, track)))


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
    plt.suptitle(f'{title} - overall accuracy: {num_correct/num_total:.4f}, mean recall: {np.diag(normalized_matrix).sum()/len(CLASSES):.4f}, mean precision: {np.sum(precision_row)/len(CLASSES):.4f}')
    plt.savefig(f'{base_path}-{title.replace(" ", "_")}test-results.png')
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
    print(model.output)
    use_certainty = isinstance(model.output, list)
    print(f'Testing model {weights_path} with input sample dimensions {sample_dims}{" and certainty output" if use_certainty else ""}:')
    class_to_index = { c:i for i,c in enumerate(CLASSES) }
    frames_correct = 0
    frames_wrong = 0
    actuals = []
    evaluation_techniques = [
        PredictionVariation('Mean prediction', lambda p, c, t: p.sum(axis=0)),
        PredictionVariation('Mean squared prediction', lambda p, c, t: (p**2).sum(axis=0)),
        PredictionVariation('Pixels weighted mean squared prediction', lambda p, c, t: sum_weighted(p**2, np.array(t.pixels))),
        PredictionVariation('Mass weighted mean squared prediction', lambda p, c, t: sum_weighted(p**2, np.array(t.masses)))
    ]
    if use_certainty:
        evaluation_techniques = evaluation_techniques + [
            PredictionVariation('Certainty weighted mean prediction', lambda p, c, t: sum_weighted(p, c)),
            PredictionVariation('Certainty weighted mean squared prediction', lambda p, c, t: sum_weighted(p**2, c))
        ]
    for track in tracks:
        tag = TAG_CLASS_MAP[track.tag]
        actual = class_to_index[tag]
        actuals.append(class_to_index[tag])
        samples = prep_track(track, sample_dims)
        if use_certainty:
            predicts, certainties = model.predict(samples)
        else:
            predicts = model.predict(samples)
            certainties = None
        for technique in evaluation_techniques:
            technique.evaluate(predicts, certainties, track)
        frame_maxes = np.argmax(predicts, axis=1)
        corrects = frame_maxes == class_to_index[tag]
        num_correct = np.sum(corrects)
        num_wrong = len(frame_maxes) - num_correct
        frames_correct += num_correct
        frames_wrong += num_wrong
        print(f'Frame predictions {num_correct / len(frame_maxes):.4f} correct and {num_wrong / len(frame_maxes):.3f} wrong ({num_correct} vs {num_wrong})')
        matches = np.array([t.accumulated_results[-1] for t in evaluation_techniques]) == actual
        if all(matches):
            print(f'Tag {tag} all correct')
        else:
            text = 'Tag ' + tag + (' partial correct' if any(matches) else ' all wrong') + ': '
            for technique in evaluation_techniques:
                text += technique.description + ' ' + CLASSES[technique.accumulated_results[-1]] + ', '
            print(text[:-2])
    total_frames = frames_correct + frames_wrong
    print(f'frames {frames_correct} predicted correctly ({frames_correct/total_frames:.4f}), {frames_wrong} predicted incorrectly ({frames_wrong/total_frames:.4f})')
    figure_path = weights_path if weights_path else model_path
    for technique in evaluation_techniques:
        evaluate(technique.accumulated_results, actuals, figure_path, technique.description.lower())


def frame_count(tracks):
    return np.sum([t.frame_count for t in tracks])


def all_frame_counts(tag_tracks):
    return np.sum([frame_count(tag_tracks[t]) for t in CLASSES])


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    argv = sys.argv
    data_directory = argv[1]
    tracks = load_raw_tracks(f'{data_directory}/test_infos.pk')
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

    load_tracks(test_tracks, f'{data_directory}/test_frames.npy')
    print(f'loaded {len(test_tracks)} tracks with {np.sum([t.frame_count for t in test_tracks])} frames')
    for weights_path in argv[2:]:
        if weights_path.endswith('.index'):
            weights_path = weights_path[:-6]
        if weights_path.startswith('fish://dennis@125.236.230.94:2040') or weights_path.startswith('sftp://dennis@125.236.230.94:2040'):
            weights_path = weights_path[33:]
        test(test_tracks, weights_path)


if __name__ == '__main__':
    sys.exit(main())
