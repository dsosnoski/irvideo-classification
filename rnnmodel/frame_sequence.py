
import math
import numpy as np
import random
from tensorflow.keras.utils import Sequence

from support.data_model import TAG_HOTS


class RnnFrameSequence(Sequence):
    """
    Sequence for samples made up of data for consecutive frames.
    """
    def __init__(self, tracks, batch_size, input_shape, noise_scale=None, classification_limit=None, classification_noise=None):
        self.tracks = tracks
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.frames_per_sample = input_shape[0]
        self.noise_scale = noise_scale
        self.classification_limit = classification_limit
        self.classification_noise = classification_noise
        # oversample longer tracks
        sample_indices = []
        for i, track in enumerate(tracks):
            if self.frames_per_sample is None:
                sample_indices.append(i)
            elif track.frame_count < self.frames_per_sample * 2:
                sample_indices.append(i)
            else:
                count = int(math.ceil(math.sqrt(track.frame_count / self.frames_per_sample)))
                sample_indices += [i] * count
        self.sample_indices = np.array(sample_indices)
        self.epoch_number = -1
        self.on_epoch_end()

    def __len__(self):
        return len(self.sample_indices) // self.batch_size

    def _prepare_sample(self, track):
        if self.frames_per_sample is None:
            values = track.data
        elif track.frame_count > self.frames_per_sample:
            start_frame = random.randrange(track.frame_count - self.frames_per_sample)
            values = track.data[start_frame:start_frame+self.frames_per_sample]
        else:
            values = []
            for index in range(self.frames_per_sample):
                if index < track.frame_count:
                    values.append(track.data[index])
                else:
                    values.append(track.data[-1])
            values = np.array(values)
        if self.noise_scale is not None:
            noise = (np.random.default_rng().random(values.shape) - .5) * self.noise_scale * 2
            sum = values + noise
            sum[(sum < 0) & (values > 0)] = 0
            sum[(sum > 0) & (values < 0)] = 0
            values = sum
        if self.classification_limit is not None:
            noise = (np.random.default_rng().random((len(values), self.classification_limit)) - .5) * self.classification_noise * 2
            sum = values + np.pad(noise, ((0, 0), (0, values.shape[1]-self.classification_limit)))
            values = np.clip(sum, 0, 1)
        return values

    def on_epoch_end(self):
        np.random.default_rng().shuffle(self.sample_indices)
        self.epoch_number += 1

    def __getitem__(self, idx):
        indexes = self.sample_indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        tracks = [self.tracks[i] for i in indexes]
        samples = [self._prepare_sample(t) for t in tracks]
        actuals = [TAG_HOTS[t.tag] for t in tracks]
        return samples, actuals
