
import sys
import traceback

import cv2
import h5py
import matplotlib
import matplotlib.cm
import numpy as np
from PySide2.QtCore import Slot
from PySide2.QtGui import QPixmap, QImage, qRgb
from PySide2.QtWidgets import QApplication, QComboBox, QHBoxLayout, QMainWindow, QPushButton, QVBoxLayout, QWidget, \
    QLabel, QStatusBar

from preprocess.image_sizing import IMAGE_DIMENSION
from support.track_utils import extract_hdf5_frames, convert_frames, clip_and_scale, extract_hdf5_crops, FRAME_DIMS, \
    normalize, prune_frames, smooth_resizer

matplotlib.use('Qt5Agg')


DATASET_PATH = '/home/dennis/projects/irvideos/working-data/problem-clips.hdf5'
CLIP_DIMS = [IMAGE_DIMENSION, IMAGE_DIMENSION]


class ViewTrack:

    def __init__(self, clip_id, track_id, background, track_hdf5):
        self.track_id = track_id
        self.background = background
        self.tag = track_hdf5.attrs['tag']
        frames = extract_hdf5_frames(track_hdf5['original'])
        self.camera_frames = frames
        self.difference_frames = self.camera_frames - background
        bounds = track_hdf5.attrs['bounds_history']
        self.raw_bounds = bounds
        crops = extract_hdf5_crops(track_hdf5['cropped'])
        self.raw_crops = crops
        masses = track_hdf5.attrs['mass_history']
        ocrops, oframes, obounds, omasses, _ = prune_frames(crops, frames, bounds, masses, clip_id, track_id)
        ratiofn = smooth_resizer(IMAGE_DIMENSION)
        aframes, obounds, ratios = convert_frames(frames, background, bounds, IMAGE_DIMENSION, ratiofn)
        self.adjust_crops = ocrops
        self.adjust_frames = aframes
        self.frame_bounds = obounds
        self.adjust_masses = omasses
        self.ratios = ratios
        self.current_index = 0

    def current_data(self):
        index = self.current_index
        print(f'returning acrop with shape {self.adjust_crops[index].shape}')
        return self.camera_frames[index], self.difference_frames[index], self.raw_crops[index], self.adjust_crops[index], self.adjust_frames[index], self.frame_bounds[index], self.adjust_masses[index]

    def next(self):
        if self.has_next():
            self.current_index += 1
        return self.current_data()

    def last(self):
        if self.has_last():
            self.current_index -= 1
        return self.current_data()

    def has_next(self):
        return self.current_index < len(self.adjust_frames) - 1

    def has_last(self):
        return self.current_index > 0



class ViewClip:

    def __init__(self, clip_id, clip_hdf5):
        self.clip_id = clip_id
        self.device_name = clip_hdf5.attrs.get('device')
        self.frames_per_second = clip_hdf5.attrs['frames_per_second']
        self.start_time = clip_hdf5.attrs['start_time']
        self.location = clip_hdf5.attrs.get('location')
        self.view_tracks = []
        self.background_frame = np.array(clip_hdf5['background_frame'])
        for key in clip_hdf5:
            try:
                if not key == 'background_frame':
                    self.view_tracks.append(ViewTrack(clip_id, key, self.background_frame, clip_hdf5[key]))
            except Exception:
                print(f'Exception processing {clip_id}-{key}')
                traceback.print_exc()
        self.current_index = 0

    def current_track(self):
        return self.view_tracks[self.current_index]

    def last(self):
        if self.has_last():
            self.current_index -= 1
        return self.current_track()

    def has_next(self):
        return self.current_index < len(self.view_tracks) - 1

    def has_last(self):
        return self.current_index > 0


class Navigator:

    def __init__(self, clip_keys, clips_hdf5):
        self.clip_keys = clip_keys
        self.clips_hdf5 = clips_hdf5
        self.current_index = 0
        self.current_clip = None

    def current_data(self):
        clip_key = self.clip_keys[self.current_index]
        if self.current_clip is None or not self.current_clip.clip_id == clip_key:
            self.current_clip = ViewClip(clip_key, self.clips_hdf5[clip_key])
        return self.current_clip

    def next(self):
        if self.has_next():
            self.current_index += 1
        return self.current_data()

    def last(self):
        if self.has_last():
            self.current_index -= 1
        return self.current_data()

    def has_next(self):
        return self.current_index < len(self.clip_keys) - 1

    def has_last(self):
        return self.current_index > 0


class ViewerWindow(QMainWindow):

    def __init__(self, frame_dims, clip_dims, navigator):
        super().__init__()
        self.data_navigator = navigator
        self.frame_dims = frame_dims
        self.view_window = QWidget()
        self.setCentralWidget(self.view_window)
        main_layout = QVBoxLayout()
        self.top_label = QLabel()
        main_layout.addWidget(self.top_label)
        full_frame_layout = QHBoxLayout()
        full_frame_layout.addStretch()
        self.frame_display_dims = tuple([d * 2 for d in reversed(frame_dims)])
        w, h = self.frame_display_dims
        self.cframe_label = self._add_label(h, w, full_frame_layout)
        self.dframe_label = self._add_label(h, w, full_frame_layout)
        self.bframe_label = self._add_label(h, w, full_frame_layout)
        full_frame_layout.addStretch()
        main_layout.addLayout(full_frame_layout)
        clips_layout = QHBoxLayout()
        main_layout.addLayout(clips_layout)
        clips_layout.addStretch()
        self.clip_display_dims = tuple([d * 5 for d in reversed(clip_dims)])
        h, w = self.clip_display_dims
        self.rcrop_label = self._add_label(h, w, clips_layout)
        self.acrop_label = self._add_label(h, w, clips_layout)
        self.xcrop_label = self._add_label(h, w, clips_layout)
        clips_layout.addStretch()
        self.frame_label = QLabel()
        main_layout.addWidget(self.frame_label)
        display_controls_layout = QHBoxLayout()
        main_layout.addLayout(display_controls_layout)
        self.last_view_button = self._add_button('<<', 120, display_controls_layout)
        self.last_view_button.clicked.connect(self._last_view_button)
        self.track_combo_box = self._add_combo_box([], 120, display_controls_layout)
        self.last_frame_button = self._add_button('<', 120, display_controls_layout)
        self.last_frame_button.clicked.connect(self._last_frame_button)
        self.next_frame_button = self._add_button('>', 120, display_controls_layout)
        self.next_frame_button.clicked.connect(self._next_frame_button)
        self.next_view_button = self._add_button('>>', 120, display_controls_layout)
        self.next_view_button.clicked.connect(self._next_view_button)
        display_controls_layout.addStretch()
        self.status_bar = QStatusBar()
        self.status_bar.showMessage('Initializing...')
        main_layout.addWidget(self.status_bar)
        self.view_window.setLayout(main_layout)
        self.set_clip(navigator.current_data)

    @Slot()
    def _next_view_button(self):
        self.set_clip(self.data_navigator.next)

    @Slot()
    def _last_view_button(self):
        self.set_clip(self.data_navigator.last)

    @Slot()
    def _next_frame_button(self):
        self.set_frame(self.view_track.next)

    @Slot()
    def _last_frame_button(self):
        self.set_frame(self.view_track.last)

    @Slot()
    def _track_combo(self):
        track_id = self.track_combo_box.currentText()
        view_track = [t for t in self.view_clip.view_tracks if t.track_id == track_id][0]
        self.set_track(view_track)

    def _add_button(self, text, maxwidth, layout):
        layout.addSpacing(40)
        button = QPushButton(text)
        button.setFixedWidth(maxwidth)
        layout.addWidget(button)
        #button.setStyleSheet('background-color: AliceBlue;')
        return button

    def _add_combo_box(self, choices, maxwidth, layout):
        layout.addSpacing(40)
        combo_box = QComboBox()
        combo_box.addItems(choices)
        combo_box.setFixedWidth(maxwidth)
        layout.addWidget(combo_box)
        return combo_box

    def _add_label(self, h, w, layout):
        layout.addSpacing(40)
        label = QLabel()
        print(f'setting label height {h}, width {w}')
        label.setFixedSize(w, h)
        layout.addWidget(label)
        return label

    def _set_clip_track_label(self):
        clip = self.view_clip
        location = clip.location if clip.location is not None else 'unknown location'
        track = self.view_track
        self.top_label.setText(f'Clip {clip.clip_id} recorded at {location} starting at {clip.start_time}: Track {track.track_id} identified as {track.tag} with {len(track.raw_bounds)} frames')

    def set_clip(self, clipfn):
        self.status_bar.showMessage('Loading clip...')
        view_clip = clipfn()
        self.view_clip = view_clip
        self.last_view_button.setEnabled(self.data_navigator.has_last())
        self.next_view_button.setEnabled(self.data_navigator.has_next())
        self.track_combo_box.currentTextChanged.connect(None)
        for i in range(self.track_combo_box.count()):
            self.track_combo_box.removeItem(i)
        track_ids = [t.track_id for t in view_clip.view_tracks]
        self.track_combo_box.addItems(track_ids)
        self.track_combo_box.currentTextChanged.connect(self._track_combo)
        self.set_track(view_clip.view_tracks[0])
        self.status_bar.clearMessage()

    def set_track(self, view_track):
        self.status_bar.showMessage(f'Setting track {view_track.track_id}')
        self.view_track = view_track
        self.set_frame(view_track.current_data)
        self.status_bar.showMessage(f'Set track {view_track.track_id}', 2000)
        self._set_clip_track_label()

    def _draw_to_label(self, data, dims, label):
        data = cv2.resize(data, dims, interpolation=cv2.INTER_CUBIC)
        h, w = data.shape
        image = QImage(w, h, QImage.Format_RGB32)
        frame = normalize(data).astype(np.uint8)
        raw_values = matplotlib.cm.magma(frame)
        #print(f'magma values min {raw_values.min()}, max {raw_values.max()}')
        color_values = np.uint8(raw_values * 255)
        for rnum in range(h):
            for cnum in range(w):
                values = color_values[rnum, cnum]
                image.setPixel(cnum, rnum, qRgb(values[0], values[1], values[2]))
        label.setPixmap(QPixmap.fromImage(image))
        label.update()

    def set_frame(self, framefn):
        cframe, dframe, rcrop, acrop, aframe, bounds, amass = framefn()
        track = self.view_track
        index = track.current_index
        pixel_count = np.sum(track.raw_crops[index] > 0)
        raw_bounds = track.raw_bounds[index]
        bound_dims = (raw_bounds[2]-raw_bounds[0], raw_bounds[3]-raw_bounds[1])
        self.frame_label.setText(f'Frame {index} : mass {track.adjust_masses[index]} pixel count {pixel_count} raw bounds {raw_bounds} dimension {bound_dims}')
        self.last_frame_button.setEnabled(self.view_track.has_last())
        self.next_frame_button.setEnabled(self.view_track.has_next())
        for frame, label in [(cframe, self.cframe_label), (dframe, self.dframe_label), (self.view_track.background, self.bframe_label)]:
            #print(f'drawing image of size {frame.shape} from track - min {frame.min()}, max {frame.max()}')
            frame = clip_and_scale(frame)
            #print(f'after clipping and scaling min {frame.min()}, max {frame.max()}')
            self._draw_to_label(frame, self.frame_display_dims, label)
        rcrop = clip_and_scale(rcrop)
        display_dims = tuple([d * 5 for d in reversed(rcrop.shape)])
        self.rcrop_label.setFixedSize(display_dims[0], display_dims[1])
        self._draw_to_label(rcrop, display_dims, self.rcrop_label)
        for crop, label in [(acrop, self.acrop_label), (aframe, self.xcrop_label)]:
            crop = clip_and_scale(crop)
            self._draw_to_label(crop, self.clip_display_dims, label)


def run_ui():
    file_hdf5 = h5py.File(DATASET_PATH, 'r')
    clips_hdf5 = file_hdf5['clips']
    app = QApplication()
    app.setApplicationName('Cacophony Infrared Video Viewer')
    clip_keys = [k for k in clips_hdf5]
    navigator = Navigator(clip_keys, clips_hdf5)
    window = ViewerWindow(FRAME_DIMS, CLIP_DIMS, navigator)
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_ui()