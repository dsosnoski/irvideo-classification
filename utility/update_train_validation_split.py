
import pickle

from model.training_utils import print_tag_track_info, load_raw_tracks, tracks_by_tag


def track_id_set(tag_tracks):
    track_ids = set()
    for tag in tag_tracks:
        tag_set = {(t.clip_key, t.track_key) for t in tag_tracks[tag]}
        track_ids = track_ids | tag_set
    return track_ids


tracks = []
with open('/data/dennis/irvideo/redata//track_split.pk', 'rb') as f:
    training_tracks = pickle.load(f)
    validation_tracks = pickle.load(f)
print('\nOriginal training data:')
print_tag_track_info(training_tracks)
print('\nOriginal validation data:')
print_tag_track_info(validation_tracks)
raw_tracks = load_raw_tracks('/data/dennis/irvideo/data/train_infos.pk')
print(f'\nLoaded {len(raw_tracks)} tracks of new data')
training_track_ids = track_id_set(training_tracks)
print(len(training_track_ids))
training_tracks = tracks_by_tag([t for t in raw_tracks if (t.clip_key, t.track_key) in training_track_ids])
validation_track_ids = track_id_set(validation_tracks)
validation_tracks = tracks_by_tag([t for t in raw_tracks if (t.clip_key, t.track_key) in validation_track_ids])
print('\nNew training data:')
print_tag_track_info(training_tracks)
print('\nNew validation data:')
print_tag_track_info(validation_tracks)
with open('/data/dennis/irvideo/data/track_split.pk', 'wb') as f:
    pickle.dump(training_tracks, f)
    pickle.dump(validation_tracks, f)
