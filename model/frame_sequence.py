
import cv2
import math
import numpy as np
import random
import time
from tensorflow.keras.utils import Sequence

from support.data_model import CLASSES

VALIDATION_SAMPLE_INTERVAL = 3
NOISE_SHAPE = (600, 600)


class TagStore:

    def __init__(self, tag, tracks, data_path, input_shape, noise_shape, noise_scale, rotation_limit, use_flip):
        self.tag = tag
        assert len(tracks) > 0, f'no tracks provided for tag {tag}'
        self.tracks = tracks
        self.data_path = data_path
        self.input_shape = input_shape
        self.noise_shape = noise_shape
        self.noise_scale = noise_scale
        self.noise_mix = None
        self.rotation_limit = rotation_limit
        self.use_flip = use_flip
        self.epoch_number = 0
        self.loaded_tracks = []

    def _load_tracks(self, tracks):
        #start_time = time.perf_counter()
        with open(self.data_path, 'rb') as f:
            for track in tracks:
                assert track.data is None, f'track data already set for track {track.track_key} with tag {track.tag}'
                assert track.offset == f.seek(track.offset)
                track.data = np.load(f)
                assert track.frame_count == len(track.data)
        #print(f'Loaded {len(tracks)} tracks with {np.sum([t.frame_count for t in tracks])} frames for tag {self.tag} in {time.perf_counter()-start_time:.3f} seconds elapsed time')

    @staticmethod
    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _format_frame(self, track, frame_index):
        h, w, d = self.input_shape
        if d == 1:
            x = np.array([track.data[frame_index].astype(np.float32)])
        else:

            # find range of frames to be used (must be same ratio)
            indexes = [frame_index]
            for index in range(frame_index+1, min(track.frame_count, frame_index+d)):
                if track.ratios[frame_index] != track.ratios[index]:
                    break
                indexes.append(index)
            if len(indexes) < d:

                # use repeated frames to pad to count required
                ratio = d / len(indexes)
                padded_indexes = []
                sum_samples = 0
                for index in indexes:
                    sum_samples += ratio
                    while len(padded_indexes) < sum_samples:
                        padded_indexes.append(index)
                indexes = padded_indexes[:d]

            x = track.data[indexes].astype(np.float32)

        if self.noise_mix is not None:

            # mixin noise
            xh, xw = x.shape[1], x.shape[2]
            r0 = random.randrange(self.noise_mix.shape[0] - xh)
            c0 = random.randrange(self.noise_mix.shape[1] - xw)
            mixin = self.noise_mix[r0:r0+xh,c0:c0+xw]
            x = np.array([np.clip(i + mixin, 0, 255) for i in x])

        if self.rotation_limit is not None and self.rotation_limit > 0 and random.random() <= self.epoch_number / 50:

            # get rotation angle, in either direction
            rotation_angle = (random.random() * 2 - 1) * self.rotation_limit

            # calculate clipping to drop skewed parts of rotated image
            clip_size = (math.cos(math.radians(rotation_angle)) * np.array(x.shape)).astype(np.int)
            clip_h = max(clip_size[0], h)
            clip_w = max(clip_size[1], w)
            base_h = (x.shape[1] - clip_h) // 2
            base_w = (x.shape[2] - clip_w) // 2

            # rotate and clip the images
            x = np.array([self.rotate_image(i, rotation_angle)[base_h:base_h+clip_h,base_w:base_w+clip_w] for i in x])

        # compute random crop of data to input size
        r0, c0 = 0, 0
        r1, c1 = h, w
        if x.shape[1] > r1:
            offset = random.randrange(x.shape[1] - r1)
            r0 += offset
            r1 += offset
        if x.shape[2] > c1:
            offset = random.randrange(x.shape[2] - c1)
            c0 += offset
            c1 += offset
        x = np.array([i[r0:r1, c0:c1] for i in x])

        # randomly flip image horizontally
        if self.use_flip and random.random() >= .5:
            x = np.flip(x, axis=2)
        x = np.moveaxis(x, 0, -1)
        return x

    def _create_noise(self):
        if self.noise_shape is not None and self.noise_scale is not None:
            self.noise_mix = np.random.default_rng().normal(size=self.noise_shape).astype(np.float32) * self.noise_scale

    def reset(self):
        self._create_noise()
        self.epoch_number += 1

    def clear(self):
        for track in self.loaded_tracks:
            track.data = None


class TrainingTagStore(TagStore):

    def __init__(self, tag, tracks, data_path, input_shape, frame_count, replace_fraction, noise_shape=None,
                 noise_scale=None, rotation_limit=None, use_flip=False):
        super().__init__(tag, tracks, data_path, input_shape, noise_shape, noise_scale, rotation_limit, use_flip)
        self.track_weight_unit = max(np.min([t.frame_count for t in tracks]), 10) / 2
        self.frame_load_count = min(frame_count, np.sum([t.frame_count for t in tracks]))
        self.replace_fraction = replace_fraction
        self.available_tracks = self.tracks.copy()
        self.next_index = len(self.available_tracks)
        self.sample_tracks = []
        self._fill_tracks()

    def _fill_tracks(self):
        frame_count = np.sum([t.frame_count for t in self.loaded_tracks])
        load_tracks = []
        while self.frame_load_count > frame_count and len(load_tracks) < len(self.available_tracks):
            track = self.available_tracks[len(load_tracks)]
            load_tracks.append(track)
            frame_count += track.frame_count
        self.available_tracks = self.available_tracks[len(load_tracks):]
        self._load_tracks(load_tracks)
        self.loaded_tracks = self.loaded_tracks + load_tracks
        for track in load_tracks:
            self.sample_tracks = self.sample_tracks + ([track] * math.ceil(track.frame_count / self.track_weight_unit))

    def reset(self):
        if len(self.available_tracks) > 0:
            sample_count_removed = math.ceil(len(self.sample_tracks) * self.replace_fraction)
            last_remove = self.sample_tracks[sample_count_removed - 1]
            while sample_count_removed < len(self.sample_tracks) and self.sample_tracks[sample_count_removed] == last_remove:
                sample_count_removed += 1
            self.sample_tracks = self.sample_tracks[sample_count_removed:]
            track_count_removed = 0
            while True:
                track = self.loaded_tracks[track_count_removed]
                track.frames = None
                track.data = None
                track_count_removed += 1
                if track == last_remove:
                    break
            dropped_tracks = self.loaded_tracks[:track_count_removed]
            self.loaded_tracks = self.loaded_tracks[track_count_removed:]
            self._fill_tracks()
            self.available_tracks = self.available_tracks + dropped_tracks
            assert self.sample_tracks[0] == self.loaded_tracks[0], f'sample_tracks {[t.track_key for t in self.sample_tracks]}, loaded_tracks {[t.track_key for t in self.loaded_tracks]}, track_count_removed {track_count_removed}'
        return super(TrainingTagStore, self).reset()

    def next_frame(self):
        track = self.sample_tracks[random.randrange(len(self.sample_tracks))]
        frame_index = random.randrange(track.frame_count)
        return self._format_frame(track, frame_index)


class ValidationTagStore(TagStore):

    def __init__(self, tag, tracks, data_path, input_shape):
        super().__init__(tag, tracks, data_path, input_shape, None, None, None, False)
        self.track_index = None
        self.sample_index = None
        self._load_tracks(self.tracks)
        self.loaded_tracks = self.tracks
        self.reset()

    def reset(self):
        self.track_index = 0
        self.sample_index = 0
        return super(ValidationTagStore, self).reset()

    def sample_count(self):
        return np.sum([t.frame_count for t in self.loaded_tracks])

    def next_frame(self):
        while self.sample_index >= self.loaded_tracks[self.track_index].frame_count:
            self.track_index += 1
            if self.track_index >= len(self.loaded_tracks):
                self.track_index = 0
            self.sample_index = 0
        frame = self._format_frame(self.loaded_tracks[self.track_index], self.sample_index)
        self.sample_index += 1
        return frame


class FrameSequence(Sequence):

    def __init__(self, tag_stores, sample_counts, batch_size, tag_hots=None):
        self.batch_size = batch_size
        self.is_train = tag_hots is not None
        self.tag_hots = tag_hots
        self.tag_stores = tag_stores
        self.sample_tags = []
        for store, count in zip(tag_stores, sample_counts):
            self.sample_tags = self.sample_tags + [store] * count
        self.on_epoch_end()

    def __len__(self):
        return len(self.sample_tags) // self.batch_size

    def on_epoch_end(self):
        np.random.default_rng().shuffle(self.sample_tags)
        for store in self.tag_stores:
            store.reset()

    def __getitem__(self, idx):
        stores = self.sample_tags[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = np.array([s.next_frame() for s in stores])
        #print(f'FrameSequence.__getitem__ providing batch with shape {batch_x.shape}')
        if self.is_train:
            batch_y = np.array([self.tag_hots[s.tag] for s in stores])
            return batch_x, batch_y
        else:
            return batch_x

    def clear(self):
        for store in self.tag_stores:
            store.clear()


    @staticmethod
    def build_training_sequence(tag_tracks, data_path, batch_size, tag_hots, input_shape, frame_counts,
                                replace_fraction, noise_scale, rotation_limit, use_flip):
        stores = []
        sample_counts = []
        print()
        print('               Tracks   Frames  Sample  Tracks   Frames')
        print('               Loaded   Loaded   Count   Total    Total')
        for tag in CLASSES:
            tracks = tag_tracks[tag]
            total_frame_count = np.sum([t.frame_count for t in tracks])
            noise_shape = NOISE_SHAPE if noise_scale is not None else None
            store = TrainingTagStore(tag, tracks, data_path, input_shape, frame_counts[tag], replace_fraction,
                                     noise_shape, noise_scale, rotation_limit, use_flip)
            stores.append(store)
            frames_loaded = np.sum([t.frame_count for t in store.loaded_tracks])
            sample_counts.append(frames_loaded)
            print(f'{tag:12}  {len(store.loaded_tracks):>7}  {frames_loaded:>7} {sample_counts[-1]:>7} {len(tracks):>7} {total_frame_count:>8}')
        return FrameSequence(stores, sample_counts, batch_size, tag_hots)

    @staticmethod
    def build_validation_sequence(tag_tracks, data_path, batch_size, tag_hots, input_shape):
        stores = []
        sample_counts = []
        for tag in CLASSES:
            tracks = tag_tracks[tag]
            store = ValidationTagStore(tag, tracks, data_path, input_shape)
            stores.append(store)
            sample_counts.append(np.sum([t.frame_count for t in tracks]))
        return FrameSequence(stores, sample_counts, batch_size, tag_hots)
