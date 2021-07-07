
import librosa
import numpy as np
import pickle
import traceback

tracks = []
with open('/home/dennis/projects/irvideos/working-data/train_info.pk', 'rb') as f:
    try:
        while True:
            tracks.append(pickle.load(f))
    except Exception as e:
        traceback.print_exc()
print(f'loaded {len(tracks)} tracks')

track = tracks[8]
with open('/home/dennis/projects/irvideos/working-data/train_frames.npy', 'rb') as f:
    f.seek(track.offset)
    frame = np.load(f)
    print(f'{track.clip_key}-{track.track_key} tag {track.tag} recorded {track.start_time} at {track.location} with {track.frame_count} frames:')
    for index in range(track.frame_count):
        l, t, r, b = track.crop_bounds[index]
        size = (b-t, r-l)
        print(f'{index:4d} mass {track.masses[index]} pixel count {track.pixels[index]} crop bound {track.crop_bounds[index]} (size {size}) ratio {track.ratios[index]}')
        print(f' frame min {frame.min()} max {frame.max()} mean {frame.mean()} std {frame.std()}')
        dbframe = librosa.power_to_db(frame, ref=np.max)
        print(f' dbframe min {dbframe.min()} max {dbframe.max()} mean {dbframe.mean()} std {dbframe.std()}')
