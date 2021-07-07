
import numpy as np
import pickle
import traceback

from support.data_model import TAG_CLASS_MAP, CLASSES


def load_raw_tracks(path):
    tracks = []
    with open(path, 'rb') as f:
        try:
            while True:
                tracks.append(pickle.load(f))
        except Exception as e:
            traceback.print_exc()
            pass
    return tracks


def tracks_by_tag(tracks):
    tag_tracks = {t: [] for t in CLASSES}
    for track in tracks:
        if track.tag in TAG_CLASS_MAP:
            track.tag = TAG_CLASS_MAP[track.tag]
            tag_tracks[track.tag].append(track)
    return tag_tracks


def print_tag_track_info(infos):
    for k in infos:
        tracks = infos[k]
        fcount = np.sum([t.frame_count for t in tracks])
        print(f'{k}: {len(tracks)} tracks with {fcount} frames')


# def remap_tags(base_tracks):
#     tag_tracks = {}
#     for tag in base_tracks:
#         tracks = base_tracks[tag]
#         retag = TAG_CLASS_MAP.get(tag)
#         if retag == tag:
#             tag_tracks[tag] = tracks
#         elif retag is not None:
#             for track in tracks:
#                 track.tag = retag
#             if retag in tag_tracks:
#                 tag_tracks[retag] = tag_tracks[retag] + tracks
#             else:
#                 tag_tracks[retag] = tracks
#     del base_tracks
#     return tag_tracks


def split_training_validation(tag_tracks, validate_frame_counts):
    train_tracks = {}
    validate_tracks = {}
    for tag in tag_tracks.keys():
        if tag in CLASSES:
            tracks = tag_tracks[tag]
            np.random.shuffle(tracks)
            vcount = 0
            train_use = []
            validate_use = []
            for track_info in tracks:
                if vcount < validate_frame_counts[tag]:
                    validate_use.append(track_info)
                    vcount += track_info.frame_count
                else:
                    train_use.append(track_info)
            train_tracks[tag] = train_use
            validate_tracks[tag] = validate_use
    return train_tracks, validate_tracks
