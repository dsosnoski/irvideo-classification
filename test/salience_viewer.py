
import os
import sys

import cv2
import matplotlib
import matplotlib.cm
import numpy as np
from PySide2.QtCore import Slot
from PySide2.QtGui import QPixmap, QImage, qRgb
from PySide2.QtWidgets import QApplication, QComboBox, QHBoxLayout, QMainWindow, QPushButton, QVBoxLayout, QWidget, \
    QLabel, QStatusBar

from model.training_utils import load_raw_tracks, tracks_by_tag
from preprocess.image_sizing import IMAGE_DIMENSION
from support.data_model import CLASSES, TAG_CLASS_MAP
from support.track_utils import clip_and_scale, FRAME_DIMS, normalize
from test.model_test import load_model
from test.test_utils import load_tracks, prep_track

matplotlib.use('Qt5Agg')


CLIP_DIMS = [IMAGE_DIMENSION, IMAGE_DIMENSION]


class ViewTrack:

    def __init__(self, track, predicts, certainties, frame_maxes, corrects):
        self.track = track
        self.current_index = 0
        self.predicts = predicts
        self.certainties = certainties
        self.frame_maxes = frame_maxes
        self.corrects = corrects

    def current_data(self):
        index = self.current_index
        track = self.track
        certainty = None if self.certainties is None else self.certainties[index][0]
        return track.data[index], track.pixels[index], self.predicts[index], certainty, self.frame_maxes[index], self.corrects[index]

    def num_correct(self):
        return np.sum(self.corrects)

    def num_wrong(self):
        return self.track.frame_count - self.num_correct()

    def next(self):
        if self.has_next():
            self.current_index += 1
        return self.current_data()

    def last(self):
        if self.has_last():
            self.current_index -= 1
        return self.current_data()

    def has_next(self):
        return self.current_index < self.track.frame_count - 1

    def has_last(self):
        return self.current_index > 0


class Navigator:

    def __init__(self, tracks, model):
        self.tracks = tracks
        self.model = model
        self.sample_dims = tuple([d for d in self.model.layers[0].output.shape[1:]])
        self.use_certainty = isinstance(self.model.output, list)
        self.current_index = 0
        self.view_tracks = []

    def current_data(self):
        if len(self.view_tracks) <= self.current_index:
            track = self.tracks[self.current_index]
            use_certainty = isinstance(self.model.output, list)
            class_to_index = {c: i for i, c in enumerate(CLASSES)}
            tag = TAG_CLASS_MAP[track.tag]
            x = prep_track(track, self.sample_dims)
            if self.use_certainty:
                predicts, certainties = self.model.predict(x)
            else:
                predicts = self.model.predict(x)
                certainties = None
            frame_maxes = np.argmax(predicts, axis=1)
            corrects = frame_maxes == class_to_index[tag]
            self.view_tracks.append(ViewTrack(track, predicts, certainties, frame_maxes, corrects))
        return self.view_tracks[self.current_index]

    def next(self):
        if self.has_next():
            self.current_index += 1
        return self.current_data()

    def last(self):
        if self.has_last():
            self.current_index -= 1
        return self.current_data()

    def has_next(self):
        return self.current_index < len(self.tracks) - 1

    def has_last(self):
        return self.current_index > 0


class ViewerWindow(QMainWindow):

    def __init__(self, frame_dims, navigator):
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
        self.sframe_label = self._add_label(h, w, full_frame_layout)
        full_frame_layout.addStretch()
        main_layout.addLayout(full_frame_layout)
        self.frame_label = QLabel()
        main_layout.addWidget(self.frame_label)
        display_controls_layout = QHBoxLayout()
        main_layout.addLayout(display_controls_layout)
        self.last_view_button = self._add_button('<<', 120, display_controls_layout)
        self.last_view_button.clicked.connect(self._last_view_button)
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
        self.set_track(navigator.current_data)

    @Slot()
    def _next_view_button(self):
        self.set_track(self.data_navigator.next)

    @Slot()
    def _last_view_button(self):
        self.set_track(self.data_navigator.last)

    @Slot()
    def _next_frame_button(self):
        self.set_frame(self.view_track.next)

    @Slot()
    def _last_frame_button(self):
        self.set_frame(self.view_track.last)

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
        label.setFixedSize(w, h)
        layout.addWidget(label)
        return label

    def _set_view_track_label(self):
        view_track = self.data_navigator.current_data()
        self.top_label.setText(f'Track {view_track.track.track_key} identified as {view_track.track.tag} with {view_track.track.frame_count} frames: {view_track.num_correct()} predicted correctly, {view_track.num_wrong()} predicted incorrectly')

    def set_track(self, trackfn):
        view_track = trackfn()
        self.status_bar.showMessage(f'Setting track {view_track.track.track_key}')
        self.view_track = view_track
        self.set_frame(view_track.current_data)
        self.status_bar.showMessage(f'Set track {view_track.track.track_key}', 2000)
        self._set_view_track_label()

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
        data, pixel_count, predict, certainty, max, correct = framefn()
        track = self.view_track
        correct_text = 'correct' if correct else 'wrong'
        self.frame_label.setText(f'Frame {track.current_index}: max {CLASSES[max]} ({predict[max]:.4f}) with certainty {certainty:.4f} ' + correct_text)
        self.last_frame_button.setEnabled(self.view_track.has_last())
        self.next_frame_button.setEnabled(self.view_track.has_next())
        self._draw_to_label(data, self.frame_display_dims, self.cframe_label)


def run_ui():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    argv = sys.argv
    data_directory = argv[1]
    tracks = load_raw_tracks(f'{data_directory}/test_infos.pk')
    weights_path = argv[2]
    tag_tracks = tracks_by_tag(tracks)
    test_tracks = []
    for tag in tag_tracks.keys():
        test_tracks = test_tracks + tag_tracks[tag]
    track_set = {(t.clip_key, t.track_key): t for t in test_tracks}
    cliplist_path = argv[3]
    with open(cliplist_path, 'r') as f:
        clip_lines = f.readlines()
    test_tracks = [track_set[tuple(i.strip().split('-'))] for i in clip_lines if len(i) > 1]
    if not test_tracks:
        print('No tracks specified')
        return
    load_tracks(test_tracks, f'{data_directory}/test_frames.npy')
    print(f'loaded {len(test_tracks)} tracks with {np.sum([t.frame_count for t in test_tracks])} frames')
    model = load_model(weights_path)
    app = QApplication()
    app.setApplicationName('Cacophony Infrared Video Viewer')
    navigator = Navigator(test_tracks, model)
    window = ViewerWindow(FRAME_DIMS, navigator)
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_ui()