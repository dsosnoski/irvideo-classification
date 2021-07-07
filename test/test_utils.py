
import numpy as np


def load_tracks(tracks, data_path):
    with open(data_path, 'rb') as f:
        for track in tracks:
            assert track.data is None
            f.seek(track.offset)
            track.data = np.load(f)
            assert track.frame_count == len(track.data)


def prep_track(track, input_shape):
    frame_samples = []
    for frame_index in range(track.frame_count):
        adjust = track.data[frame_index].astype(np.float32)
        h, w, _ = input_shape
        r0, c0 = 0, 0
        r1, c1 = h, w
        if adjust.shape[0] > r1:
            offset = (adjust.shape[0] - r1) // 2
            r0 += offset
            r1 += offset
        if adjust.shape[1] > c1:
            offset = (adjust.shape[1] - c1) // 2
            c0 += offset
            c1 += offset
        adjust = adjust[r0:r1, c0:c1]
        if input_shape[2] == 1:
            frame_samples.append(adjust)
        else:
            frame_samples.append(np.repeat(adjust[..., np.newaxis], input_shape[2], -1))
    return np.array(frame_samples)
