import os
import shutil

src_path = '/input/Experiment/data/test'
dest_path = '/input/Experiment/proc'
prefix = 'vox_eval%'

vids = os.listdir(src_path)

for vid in vids:
    vid_path = os.path.join(src_path, vid)
    tgt_path = os.path.join(dest_path, prefix + vid)
    shutil.copytree(vid_path, tgt_path)
    print(f'copying {tgt_path} done')