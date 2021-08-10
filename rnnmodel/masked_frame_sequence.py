
import math
import numpy as np
import random

from tensorflow.keras.utils import Sequence

from support.data_model import TAG_HOTS


class RnnFrameSequence(Sequence):
    """
    Sequence for samples made up of data for consecutive frames.
    """
    def __init__(self, tracks, batch_size, frames_per_sample, per_frame_inputs, noise_scales=None):
        """
        Initialize frame sequence for one or more inputs. The actual data must be supplied for each track, in the form
        of a list of inputs values ([input1, input2, ...]). Inputs consisting of frame data are two-dimensional numpy
        arrays, with one row per frame; inputs of track data are one-dimensional numpy arrays.
        :param tracks: training tracks
        :param batch_size:  batch size
        :param frames_per_sample: number of frames to use for each sample (if greater than count in track, values are
         zero padded from start; if less than count in track, starting frame index is random within range)
        :param per_frame_inputs: True if input is per frame, False if input per track, for each input
        :param noise_scales: tuple (scale, clip low, clip high, per frame) for each input, scale specifies noise range
          [-scale, scale], clip low/high give clipping range, and per frame says whether to use separate randoms for
          each frame or a single set for all
        """
        self.tracks = tracks
        self.batch_size = batch_size
        self.frames_per_sample = frames_per_sample
        self.per_frame_inputs = per_frame_inputs
        self.noise_scales = noise_scales if noise_scales is not None else [None for _ in per_frame_inputs]
        # oversample longer tracks
        sample_indices = []
        for i, track in enumerate(tracks):
            if track.frame_count < self.frames_per_sample * 2:
                sample_indices.append(i)
            else:
                count = int(math.ceil(math.sqrt(track.frame_count / self.frames_per_sample)))
                sample_indices += [i] * count
        self.sample_indices = np.array(sample_indices)
        self.epoch_number = -1
        self.randgen = np.random.default_rng()
        self.on_epoch_end()

    def __len__(self):
        return len(self.sample_indices) // self.batch_size

    def _noise_and_pad(self, data, per_frame_input, start_index, end_index, noise_spec):
        if per_frame_input:
            data = data[start_index:end_index].astype(np.float32)
            if noise_spec is not None:
                scale, clip_low, clip_high, per_frame_noise = noise_spec
                if per_frame_noise:
                    noise = (self.randgen.random(data.shape, dtype=np.float32) - .5) * scale * 2
                else:
                    noise = (self.randgen.random(data.shape[1:], dtype=np.float32) - .5) * scale * 2
                data += noise
                data = np.clip(data, clip_low, clip_high)
            pad_count = self.frames_per_sample - (end_index - start_index)
            if pad_count > 0:
                data = np.pad(data, ((0, pad_count),(0,0)))
        elif noise_spec is not None:
            scale, clip_low, clip_high, _ = noise_spec
            noise = (self.randgen.random(data.shape, dtype=np.float32) - .5) * scale * 2
            data += noise
            data = np.clip(data, clip_low, clip_high)
        return data

    def _prepare_sample(self, track):
        if track.frame_count > self.frames_per_sample:
            start_index = self.randgen.integers(0, track.frame_count - self.frames_per_sample)
            end_index = start_index+self.frames_per_sample
        else:
            start_index = 0
            end_index = track.frame_count
        values = []
        for input_data, per_frame, noise_spec in zip(track.data, self.per_frame_inputs, self.noise_scales):
            values.append(self._noise_and_pad(input_data, per_frame, start_index, end_index, noise_spec))
        return values

    def on_epoch_end(self):
        self.randgen.shuffle(self.sample_indices)
        self.epoch_number += 1

    def _pad_to_length(self, sample, length):
        if length > len(sample):
            sample = np.pad(sample, ((0, length - len(sample)), (0, 0)))
        return sample

    def _print_sample(self, sample):
        for index, row in enumerate(sample):
            text = f'{index:>5}: '
            for col in row:
                text += f' {col:.4f}'
            print(text)

    def __getitem__(self, idx):
        indexes = self.sample_indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        tracks = [self.tracks[i] for i in indexes]
        inputs = [[] for _ in self.per_frame_inputs]
        #self.randgen = np.random.default_rng(42)
        for track in tracks:
            for input_list, value in zip(inputs, self._prepare_sample(track)):
                input_list.append(value)
        inputs = [np.array(s) for s in inputs]
        # if idx == 0 and self.epoch_number == 0:
        #     index = 0
        #     while index < 16:
        #         print(f'Sample is track {tracks[index].track_key} of length {tracks[index].frame_count} at index {indexes[index]}')
        #         print(f'input0:')
        #         self._print_sample(inputs[0][index])
        #         print(f'input1:')
        #         self._print_sample(inputs[1][index])
        #         index += 1
        actuals = np.array([TAG_HOTS[t.tag] for t in tracks])
        return tuple(inputs), actuals
