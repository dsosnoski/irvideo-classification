
import numpy as np


CLASSES = ['bird', 'cat', 'hedgehog', 'human', 'leporidae', 'mustelid', 'possum', 'rodent', 'unclassified', 'vehicle',
           'wallaby']
UNCLASSIFIED_TAGS = {'false-positive', 'insect'}
TAG_CLASS_MAP = {k: k for k in CLASSES}
TAG_CLASS_MAP.update({k: 'unclassified' for k in UNCLASSIFIED_TAGS})
TAG_INDEXES = {k:i for i,k in enumerate(CLASSES)}
TAG_HOTS = {}
for index, tag in enumerate(CLASSES):
    one_hot = np.zeros(len(CLASSES))
    one_hot[index] = 1
    TAG_HOTS[tag] = one_hot


class Track:

    def __init__(self, tag, clip_id, track_id, start_time, device_name, location, masses, pixels, crop_bounds, adjusted_bounds, ratios, offset=None, data=None):
        self.tag = tag
        self.clip_key = clip_id
        self.track_key = track_id
        self.start_time = start_time
        self.device_name = device_name
        self.location = location
        self.masses = masses
        self.pixels = pixels
        self.frame_count = len(crop_bounds)
        self.crop_bounds = crop_bounds
        self.adjusted_bounds = adjusted_bounds
        self.ratios = ratios
        self.offset = offset
        self.data = data

