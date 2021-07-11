import sys

import h5py


def copy_group(igroup, ogroup):
    for attr_name in igroup.attrs:
        ogroup.attrs[attr_name] = igroup.attrs[attr_name]
    for sub_name in igroup:
        sub_thing = igroup[sub_name]
        if type(sub_thing) == h5py._hl.group.Group:
            copy_group(sub_thing, ogroup.create_group(sub_name))
        elif type(sub_thing) == h5py._hl.dataset.Dataset:
            ogroup.create_dataset(sub_name, data=sub_thing)


def main():
    argv = sys.argv
    input_path = argv[1]
    cliplist_path = argv[2]
    output_path = argv[3]
    ifile = h5py.File(input_path, 'r')
    ofile = h5py.File(output_path, 'w')
    with open(cliplist_path, 'r') as f:
        ids = f.readlines()
    clip_ids = list({i.split('-')[0] for i in ids if len(i) > 1})
    in_clips = ifile['clips']
    out_clips = ofile.create_group('clips')
    for clip_id in clip_ids:
        copy_group(in_clips[clip_id], out_clips.create_group(clip_id))
    ifile.close()
    ofile.close()


if __name__ == '__main__':
    sys.exit(main())
