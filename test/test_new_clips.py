import traceback

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sn
import sys
import tensorflow as tf


from support.data_model import CLASSES, TAG_CLASS_MAP, Track, UNCLASSIFIED_TAGS
from support.track_utils import convert_frames, convert_hdf5_frames


START_TIME = '2021-03-29T08:07:54.240643+13:00'

sample_dims = (32,32)
bright_pixel_threshold = .25


def format_sample(dims, input):
    def slice(actual, limit):
        start = (limit - actual) // 2
        end = start + actual
        return start, end

    sample = np.zeros(dims, np.float32)
    row0, row1 = slice(input.shape[0], dims[0])
    col0, col1 = slice(input.shape[1], dims[1])
    sample[row0:row1, col0:col1] = input
    sample = sample.reshape(sample.shape + (1,))
    return sample


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
    plt.savefig(f'{base_path}-{title.replace(" ", "_")}test-results.png')
    plt.close()


def sum_weighted(predicts, weights):
    return np.matmul(weights.T, predicts)


def test(tracks, model_path, weights_path=None):
    model = tf.keras.models.load_model(model_path)
    if weights_path:
        model.load_weights(weights_path)
    print(f'Testing model {model_path} with weights {weights_path}:')
    class_to_index = { c:i for i,c in enumerate(CLASSES) }
    frames_correct = 0
    frames_wrong = 0
    actuals = []
    frame_mean_predicts = []
    frame_squared_predicts = []
    frame_pixelcount_predicts = []
    frame_pixelcount_sqrpredicts = []
    for track in tracks:
        tag = track.tag
        if tag in UNCLASSIFIED_TAGS:
            tag = 'unclassified'
        actuals.append(class_to_index[tag])
        frames = track.frames
        frames = np.array([format_sample(sample_dims, f) for f in frames])
        predicts = model.predict(frames)
        frame_mean_predicts.append(np.argmax(predicts.sum(axis=0)))
        predicts_squared = predicts**2
        frame_squared_predicts.append(np.argmax(predicts_squared.sum(axis=0)))
        pixelcount_weights = np.array([(f > 0).sum() for f in frames])
        frame_pixelcount_predicts.append(np.argmax(sum_weighted(predicts, pixelcount_weights)))
        frame_pixelcount_sqrpredicts.append(np.argmax(sum_weighted(predicts_squared, pixelcount_weights)))
        frame_maxes = np.argmax(predicts, axis=1)
        correct = np.sum(frame_maxes == class_to_index[tag])
        frames_correct += correct
        frames_wrong += len(frame_maxes) - correct
        print(f'{track.clip_key}-{track.track_key} with tag {tag} has {correct} correct and {len(frame_maxes) - correct} wrong predictions')
    total_frames = frames_correct + frames_wrong
    print(f'frames {frames_correct} predicted correctly ({frames_correct/total_frames:.4f}), {frames_wrong} predicted incorrectly ({frames_wrong/total_frames:.4f})')
    figure_path = weights_path if weights_path else model_path
    evaluate(frame_mean_predicts, actuals, figure_path, 'frame mean')
    evaluate(frame_squared_predicts, actuals, figure_path, 'frame squared predicts mean')
    evaluate(frame_pixelcount_predicts, actuals, figure_path, 'nonzero pixel count weighting')
    evaluate(frame_pixelcount_sqrpredicts, actuals, figure_path, 'nonzero pixel count weighting squared predicts')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    argv = sys.argv
    dataset_path = argv[1]
    model_directory = argv[2]
    file_hdf5 = h5py.File(dataset_path, 'r')
    clips_hdf5 = file_hdf5['clips']
    tracks = []
    for clip_id in clips_hdf5:
        clip_hdf5 = clips_hdf5[clip_id]
        #if clip_hdf5.attrs['start_time'] > START_TIME:
        if clip_hdf5.attrs['start_time'] < '2019-10-29T08:07:54.240643+13:00':
            tkeys = [k for k in clip_hdf5 if not k == 'background_frame']
            for track_id in tkeys:
                try:
                    track_hdf5 = clip_hdf5[track_id]
                    tag = track_hdf5.attrs['tag']
                    frames, bounds = convert_hdf5_frames(track_hdf5['cropped'], track_hdf5.attrs['bounds_history'])
                    frames, bounds = convert_frames(frames, bounds, clip_id, track_id)
                    if len(frames):
                        start_time = track_hdf5.attrs['start_time']
                        end_time = track_hdf5.attrs['end_time']
                        if tag in TAG_CLASS_MAP:
                            tag = TAG_CLASS_MAP[tag]
                            tracks.append(Track(tag, clip_id, track_id, start_time, end_time, bounds, None, frames))
                        else:
                            print(f'Ignoring {clip_id}-{track_id}: unsupported tag {tag}')
                    else:
                        print(f'Ignoring {clip_id}-{track_id}: no usable frames')
                except Exception:
                    print(f'Exception processing {clip_id}-{track_id}')
                    traceback.print_exc()
    print(f'found {len(tracks)} with start times after {START_TIME}')
    if len(argv) > 3:
        test(tracks, f'{model_directory}/model.sav', f'{model_directory}/{argv[3]}')
    else:
        test(tracks, f'{model_directory}/model.sav')


if __name__ == '__main__':
    sys.exit(main())
