
import dateutil as du

from model.training_utils import load_raw_tracks, tracks_by_tag, print_tag_track_info

OUTPUT_DIRECTORY = '/data/dennis/irvideo/new-data'

tracks = load_raw_tracks(f'{OUTPUT_DIRECTORY}/train_infos.pk')
print(len(tracks))
last_start = du.parser.parse('1980-01-01T00:00:00+13:00')
for track in tracks:
    if last_start < track.start_time:
        last_start = track.start_time

print(last_start)