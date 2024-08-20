import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from tqdm import tqdm
import numpy as np
import joblib
import copy
import cv2
from multiprocessing import Pool
import shutil





def split_all_files(all_files, data_dir):
    for file in tqdm(all_files):
        try:
            data_entry = joblib.load(file)
        except:
            print("bad file", file)
            continue
        take_key = list(data_entry.keys())[0]
        seq_len = data_entry[take_key]['trans_orig'].shape[0]
        
        if seq_len > 1000:
            indxes = np.arange(seq_len)
            seg_length = 450
            splits = np.array_split(indxes, len(indxes) // seg_length + 1)
            
            for split in splits:
                seq_start, seq_end = split[0], split[-1]
                data_dump = {k: v[seq_start:seq_end+1] if not k in ['fps', 'scale', 'smpl_data', 'track_idx'] else v for k, v in data_entry[take_key].items()}
                dump_key = f"{take_key}_{seq_start}_{seq_end}"
                
                joblib.dump({dump_key: data_dump}, osp.join(data_dir, f"{data_split}_seg/{dump_key}.pkl"), compress = True)
                
                del data_dump['segmentation_mono']
                del data_dump['heatmaps']
                joblib.dump({dump_key: data_dump}, osp.join(data_dir, f"{data_split}_seg_motion/{dump_key}.pkl"), compress = True)
                
        else:
            joblib.dump(data_entry, osp.join(data_dir, f"{data_split}_seg/{take_key}.pkl"))
            
            del data_entry[take_key]['segmentation_mono']
            del data_dump['heatmaps']
            
            joblib.dump(data_entry, osp.join(data_dir, f"{data_split}_seg_motion/{take_key}.pkl"))


data_dir = "/hdd2/zen/data/SimXR/syn"
###################### Splitting data into Train and Test ######################
sinlges_dir = osp.join(data_dir, "singles")
train_dir = osp.join(data_dir, f"train/")
test_dir = osp.join(data_dir, f"test/")
with open("sample_data/train_syn.txt", "r") as f: train_split = f.read().split("\n")
with open("sample_data/test_syn.txt", "r") as f: test_split = f.read().split("\n")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

all_files = os.listdir(sinlges_dir)
for file in tqdm(all_files):
    if file in train_split:
        shutil.copy2(osp.join(sinlges_dir, file), osp.join(train_dir, file))
    elif file in test_split:
        shutil.copy2(osp.join(sinlges_dir, file), osp.join(test_dir, file))
        

###################### Splitting data into Segments ######################
for data_split in ["train", "test"]:
    all_files = glob.glob(osp.join(data_dir, f"{data_split}/*"))
    os.makedirs(osp.join(data_dir, f"{data_split}_seg/"), exist_ok=True)
    os.makedirs(osp.join(data_dir, f"{data_split}_seg_motion/"), exist_ok=True)


    jobs = all_files
    num_jobs = 10
    chunk = np.ceil(len(jobs)/num_jobs).astype(int)
    jobs= [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    job_args = [(jobs[i], data_dir) for i in range(len(jobs))]
    print(len(job_args))

    try:
        pool = Pool(num_jobs)   # multi-processing
        pool.starmap(split_all_files, job_args)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()