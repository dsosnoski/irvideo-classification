

CLASSES = ['bird', 'cat', 'hedgehog', 'human', 'leporidae', 'mustelid', 'possum', 'rodent', 'unclassified', 'vehicle',
           'wallaby']
UNCLASSIFIED_TAGS = {'false-positive', 'insect'}
TAG_CLASS_MAP = {k: k for k in CLASSES}
TAG_CLASS_MAP.update({k: 'unclassified' for k in UNCLASSIFIED_TAGS})


class Track:

    def __init__(self, tag, clip_key, track_key, start_time, device_name, location, masses, pixels, crop_bounds, adjusted_bounds, ratios, offset=None, data=None):
        self.tag = tag
        self.clip_key = clip_key
        self.track_key = track_key
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

