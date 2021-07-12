
import numpy as np

from model.training_utils import load_raw_tracks, tracks_by_tag, print_tag_track_info

OUTPUT_DIRECTORY = '/data/dennis/irvideo/data'

tracks = load_raw_tracks(f'{OUTPUT_DIRECTORY}/train_infos.pk')
print(len(tracks))

tag_tracks = tracks_by_tag(tracks)
print_tag_track_info(tag_tracks)