
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import tensorflow as tf

from model.f1_metric import F1ScoreMetric
from model.model_certainty import ModelWithCertainty
from support.data_model import CLASSES, TAG_CLASS_MAP
from test.prediction_variations import get_general_predictions, get_predictions_with_certainty
from test.test_utils import prep_track, load_model


class ModelTest:

    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.model = load_model(weights_path)
        self.sample_dims = tuple([d for d in self.model.layers[0].output.shape[1:]])
        self.use_certainty = isinstance(self.model.output, list)
        self.evaluation_techniques = get_predictions_with_certainty() if self.use_certainty else  get_general_predictions()

    def _evaluate(self, predicts, actuals, base_path, title):
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
        plt.suptitle(f'{title} \n overall accuracy: {num_correct/num_total:.4f}, mean recall: {np.diag(normalized_matrix).sum()/len(CLASSES):.4f}, mean precision: {np.sum(precision_row)/len(CLASSES):.4f}')
        plt.savefig(f'{base_path}-{title.replace(" ", "_")}test-results.png')
        plt.close()

    def test(self, tracks, show_cms=False):
        for e in self.evaluation_techniques:
            e. reset()
        class_to_index = { c:i for i,c in enumerate(CLASSES) }
        frames_correct = 0
        frames_wrong = 0
        actuals = []
        track_predicts = []
        track_certainties = [] if self.use_certainty else None
        for track in tracks:
            tag = TAG_CLASS_MAP[track.tag]
            actual = class_to_index[tag]
            actuals.append(class_to_index[tag])
            samples = prep_track(track, self.sample_dims)
            if self.use_certainty:
                predicts, certainties = self.model.predict(samples)
                track_certainties.append(certainties)
            else:
                predicts = self.model.predict(samples)
                certainties = None
            track_predicts.append(predicts)
            for technique in self.evaluation_techniques:
                technique.evaluate(predicts, certainties, track)
            frame_maxes = np.argmax(predicts, axis=1)
            corrects = frame_maxes == class_to_index[tag]
            num_correct = np.sum(corrects)
            num_wrong = len(frame_maxes) - num_correct
            frames_correct += num_correct
            frames_wrong += num_wrong
            #print(f'Track {track.track_key} tag {tag} frame predictions {num_correct / len(frame_maxes):.4f} correct and {num_wrong / len(frame_maxes):.4f} wrong ({num_correct} vs {num_wrong})')
            matches = np.array([t.accumulated_results[-1] for t in self.evaluation_techniques]) == actual
            if all(matches):
                #print(f'Track {track.track_key} prediction variations all correct')
                pass
            else:
                text = f'{track.clip_key}-{track.track_key} prediction variations' + (' partial correct' if any(matches) else ' all wrong') + ': '
                for technique in self.evaluation_techniques:
                    text += technique.description + ' ' + CLASSES[technique.accumulated_results[-1]] + ', '
                print(text[:-2])
        total_frames = frames_correct + frames_wrong
        print(f'frames {frames_correct} predicted correctly ({frames_correct/total_frames:.4f}), {frames_wrong} predicted incorrectly ({frames_wrong/total_frames:.4f})')
        if show_cms:
            for technique in self.evaluation_techniques:
                self._evaluate(technique.accumulated_results, actuals, self.weights_path, technique.description.lower())
        return track_predicts, track_certainties