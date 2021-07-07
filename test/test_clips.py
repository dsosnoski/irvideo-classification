
import h5py
import numpy as np
import os
import sys
import tensorflow as tf

from support.data_model import CLASSES, TAG_CLASS_MAP, Track
from support.track_utils import convert_frames, convert_hdf5_frames

sample_dims = (32,32)


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


def test(tracks, model_path, weights_path=None):
    model = tf.keras.models.load_model(model_path)
    if weights_path:
        model.load_weights(weights_path)
    print(f'Testing model {model_path} with weights {weights_path}:')
    for track in tracks:
        tag = track.tag
        print(f'Testing {track.clip_key}-{track.track_key} identified as {tag}:')
        frames = track.frames
        frames = np.array([format_sample(sample_dims, f) for f in frames])
        predicts = model.predict(frames)
        for i, p in enumerate(predicts):
            high = np.argmax(p)
            print(f' frame {i} prediction {CLASSES[high]} ({p[high]})')
        print(f'Mean prediction {CLASSES[np.argmax(predicts.sum(axis=0))]}')
        pixelcount_weights = np.array([(f > 0).sum() for f in frames])
        print(f'Weighted prediction {CLASSES[np.argmax(np.matmul(pixelcount_weights.T, predicts))]}')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    argv = sys.argv
    dataset_path = argv[1]
    model_directory = argv[2]
    weights = [n[:-6] for n in os.listdir(model_directory) if n.endswith('.index')]
    file_hdf5 = h5py.File(dataset_path, 'r')
    clips_hdf5 = file_hdf5['clips']
    tracks = []
    for clip_id in argv[3:]:
        clip_hdf5 = clips_hdf5[clip_id]
        tkeys = [k for k in clip_hdf5 if not k == 'background_frame']
        for track_id in tkeys:
            track_hdf5 = clip_hdf5[track_id]
            tag = track_hdf5.attrs['tag']
            frames, bounds = convert_hdf5_frames(track_hdf5['cropped'], track_hdf5.attrs['bounds_history'])
            frames, bounds = convert_frames(frames, bounds, clip_id, track_id)
            if len(frames):
                start_time = track_hdf5.attrs['start_time']
                end_time = track_hdf5.attrs['end_time']
                tag = TAG_CLASS_MAP[tag]
                tracks.append(Track(tag, clip_id, track_id, start_time, end_time, bounds, None, frames))
            else:
                print(f'Ignoring {clip_id}-{track_id}: no usable frames')

    if weights:
        for weight in weights:
            test(tracks, f'{model_directory}/model.sav', f'{model_directory}/{weight}')
    else:
        test(tracks, f'{model_directory}/model.sav')


if __name__ == '__main__':
    sys.exit(main())
