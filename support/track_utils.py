
import cv2
import librosa
import numpy as np

FRAME_DIMS = [120, 160]
MIN_LEFT_COLUMN = 1
MIN_TOP_ROW = 1
MAX_RIGHT_COLUMN = FRAME_DIMS[1] - 1
MAX_BOTTOM_ROW = FRAME_DIMS[0] - 1
MAX_DIMENSION = min(MAX_BOTTOM_ROW-MIN_TOP_ROW, MAX_RIGHT_COLUMN-MIN_LEFT_COLUMN)


def extract_hdf5_frames(hdf5_frames):
    """
    Extract frames from HDF5 dataset. This converts the frames to a list.
    :param hdf5_frames: original video frames
    :return [frame] list of frames
    """
    frames = []
    for i in range(len(hdf5_frames)):
        hdf5_frame = hdf5_frames[str(i)]
        assert len(hdf5_frame) == 120
        frame_rows = []
        for rnum in range(len(hdf5_frame)):
            row = hdf5_frame[rnum]
            frame_rows.append(row)
        frames.append(np.array(frame_rows))
    return frames


def extract_hdf5_crops(hdf5_crops):
    """
    Extract crops from HDF5 dataset. This converts the crops to a list.
    :param hdf5_crops: stored crops
    :return [crop] list of crops
    """
    crops = []
    for i in range(len(hdf5_crops)):
        crops.append(hdf5_crops[str(i)][1])
    return crops


def center_position(low_bound, high_bound, low_limit, high_limit, space):
    """
    Center bounds within available space.
    :param low_bound: current lower bound
    :param high_bound: current upper bound
    :param low_limit: minimum allowed bound
    :param high_limit: maximum allowed bound
    :param space: available space
    :return: centered low bound, centered high bound
    """
    size = high_bound - low_bound
    extra = space - size
    if extra > 0:
        if low_bound == low_limit:
            return 0, size
        elif high_bound == high_limit:
            return space - size, space
        else:
            leading_pad = extra // 2
            adjusted_low_bound = low_bound - leading_pad
            if adjusted_low_bound < low_limit:
                leading_pad = low_bound - low_limit
            adjusted_high_bound = low_bound - leading_pad + space
            if adjusted_high_bound > high_limit:
                leading_pad = space - size - (high_limit - high_bound)
            return leading_pad, leading_pad + size
    else:
        return 0, size


def square_bounds(bounds, size):
    l, t, r, b = bounds
    if b-t != size or r-l != size:

        # pad or crop bounds to square
        size = min(size, MAX_DIMENSION)
        dc0, dc1 = center_position(l, r, MIN_LEFT_COLUMN, MAX_RIGHT_COLUMN, size)
        dr0, dr1 = center_position(t, b, MIN_TOP_ROW, MAX_BOTTOM_ROW, size)

        # adjust values for cropping frames
        t -= dr0
        b = t + size
        l -= dc0
        r = l + size
        assert t >= MIN_TOP_ROW and b <= MAX_BOTTOM_ROW and l >= MIN_LEFT_COLUMN and r <= MAX_RIGHT_COLUMN, \
            f'square_cropped original bounds {bounds} with output bounds {(l, t, r, b)}'

    return (l, t, r, b)


def normalize(x):
    x -= x.min()
    xmax = x.max()
    if xmax > 0:
        x = x.astype(np.float32) * (255 / xmax)
    return x


def clip_and_scale(frame):
    """
    Clip values in frame to [mean - 1.5*std, mean + 3*std], and scale to 0-255 range.
    :param frame: frame
    :return: clipped and scaled frame
    """
    frame_min = np.min(frame)
    frame_max = np.max(frame)
    if frame_max > frame_min:
        frame_mean = np.mean(frame)
        frame_std = np.std(frame)
        use_min = max(frame_min, frame_mean - 3 * frame_std // 2)
        use_range = frame_max - use_min
        assert use_range > 0, f'clipping with use_min {use_min}, frame_max {frame_max}'
        frame = np.clip(frame, use_min, frame_max) - use_min
        return (frame / use_range) * 255
    else:
        return frame


def sizing(maxdim, scale_base, max_upscale):
    if maxdim > scale_base:
        resize_ratio = .5 if maxdim > scale_base * 1.65 \
            else .6 if maxdim > scale_base * 1.55 \
            else .675 if maxdim > scale_base * 1.4\
            else .75
        newmax = int(scale_base / resize_ratio)
    elif maxdim < scale_base / 2:
        resize_ratio = min(int(scale_base / maxdim), max_upscale)
        newmax = int(scale_base / resize_ratio)
    elif maxdim < scale_base / 1.5:
        resize_ratio = 1.5
        newmax = int(scale_base / resize_ratio)
    else:
        resize_ratio = 1
        newmax = int(maxdim / resize_ratio)
    return resize_ratio, newmax


def prune_frames(icrops, iframes, ibounds, imasses, clip_key, track_key, difference_to_keep=24):
    """
    Drop useless frames, based on the cropped frame data. If a crop is either all zero, or the same bounds as the prior
    crop and with no significant change in pixel values, it's dropped from the list (along with the matching frame,
    bound, and mass). For all frames retained a count of non-zero pixels in the cropped frame data is calculated and
    returned. All returned values, except for the crops and frames, are in the form of numpy arrays.

    :param icrops: dataset crops
    :param iframes: original video frame
    :param ibounds: input bounds
    :param imasses: input masses
    :param clip_key: clip identifier
    :param track_key: track identifier
    :param difference_to_keep: minimum change in crop pixel values to retain frame
    :return: (crops, adjusts, bounds, masses, pixels)
    """
    ocrops = []
    oframes = []
    obounds = []
    omasses = []
    opixels = []
    zero_count = 0
    static_count = 0
    last_crop = None
    last_bound = None
    for crop, frame, bound, mass in zip(icrops, iframes, ibounds, imasses):
        if not np.any(crop):
            zero_count += 1
        else:
            bound = np.array(bound)
            if last_crop is not None and np.array_equal(bound, last_bound) and np.sum(np.abs(crop - last_crop)) < difference_to_keep:
                static_count += 1
            else:
                last_crop = crop
                last_bound = bound
                ocrops.append(crop)
                oframes.append(frame)
                obounds.append(bound)
                omasses.append(mass)
                opixels.append((crop > 0).sum())
    if zero_count > 0 or static_count > 0:
        print(f'{clip_key}-{track_key} dropped {zero_count} zero crops and {static_count} static crops')
    return ocrops, oframes, np.array(obounds), np.array(omasses), np.array(opixels)


def stepped_resizer(resize_ratios):

    def calculate_resize_stepped(maxdim):
        return resize_ratios[maxdim]

    return calculate_resize_stepped


def smooth_resizer(sample_dim):

    def calculate_smooth_resize(maxdim):
        ratio = maxdim / sample_dim
        if ratio < 0:
            return min(1, ratio * 1.5)
        else:
            return ratio


    return calculate_smooth_resize


def convert_frames(iframes, background, ibounds, out_dim, resize_calculatorfn):
    """
    Convert input crops and raw frames to standardized form for use with a model. The largest dimension of the crop is
    used to lookup a resize ratio. The portion of frame data used as input to the resizing is centered on the crop area
    (respecting the borders of the frame), and is first adjusted by subtracting the background and normalizing.

    :param iframes: original video frame
    :param background: background frame
    :param ibounds: input bounds
    :param out_dim: output size (same dimension for both rows and columns)
    :param resize_calculatorfn: function to calculate ratio for resizing images
    :return: (aframes, obounds, ratios)
    """
    aframes = []
    obounds = []
    ratios = []
    for frame, bound in zip(iframes, ibounds):
        af = frame - background
        af *= af > 0
        l, t, r, b = bound
        maxdim = max(b-t, r-l)
        resize_ratio = resize_calculatorfn(maxdim)
        usedim = int(resize_ratio * out_dim)
        bnds = bound
        # dc0, dc1 = center_position(l, r, MIN_LEFT_COLUMN, MAX_RIGHT_COLUMN, usedim)
        # dr0, dr1 = center_position(t, b, MIN_TOP_ROW, MAX_BOTTOM_ROW, usedim)
        # expanded_crop = np.zeros((usedim,usedim), np.float32)
        # expanded_crop[dr0:dr1,dc0:dc1] = crop
        # crop = expanded_crop
        bnds = square_bounds(bnds, usedim)
        l, t, r, b = bnds
        af = clip_and_scale(af[t:b, l:r].astype(np.float32))
        if not af.max() < 255.5:
            print(f'convert_frames invalid af.max()={af.max()} after initial clip_and_scale, maxdim={maxdim}, bounds={bound}')
        resize_inter = cv2.INTER_AREA if resize_ratio > 1 else cv2.INTER_CUBIC if resize_ratio < 1 else None
        if resize_inter is not None:
            resize_dims = (out_dim, out_dim)
            #crop = cv2.resize(crop, resize_dims, interpolation=resize_inter)
            af = cv2.resize(af, resize_dims, interpolation=resize_inter)
        afscaled = normalize(librosa.power_to_db(af, ref=np.max))
        # crop = normalize(crop)
        # if not crop.max() < 255.5:
        #     print(f'convert_frames invalid crop.max()={crop.max()} after normalize, maxdim={maxdim}, bounds={bounds}')
        af = normalize(af)
        if not af.max() < 255.5:
            print(f'convert_frames invalid af.max()={af.max()} after normalize, maxdim={maxdim}, bounds={bound}')
        # crop = np.round(crop).astype(np.uint8)
        # ocrops.append(crop)
        af = np.round(af).astype(np.uint8)
        aframes.append(af)
        obounds.append(bnds)
        ratios.append(resize_ratio)
    return np.array(aframes), np.array(obounds), np.array(ratios)
