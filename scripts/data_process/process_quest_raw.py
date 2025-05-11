import io
import re
from enum import Enum

import numpy as np

import pyvrs
from PIL import Image
from collections import defaultdict
import cv2
import msgpack
import joblib
import matplotlib.pyplot as plt
import torch
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm
import glob
# from uhc.utils.image_utils import write_frames_to_video_io

# key_id = "capture01"

# device = "quest2"

# vrs_reader = pyvrs.reader.SyncVRSReader(f'data/mmt/real/{device}/raw/{key_id}.vrs')
# # for record in vrs_reader:  # loop over all records
# #     print(record)  # print all the state of the record

# filtered_reader = vrs_reader.filtered_by_fields(
#                     stream_ids = {'1201-*' }, # limit to some streams
#                   )

# stream_to_img = defaultdict(list)
# time_stamps = []
# for filtered_record in filtered_reader:
#     if len(filtered_record.image_blocks) > 0:
#         assert(len(filtered_record.image_blocks) == 1)
#         stream_to_img[filtered_record.stream_id].append(filtered_record.image_blocks[0])
#         time_stamps.append(filtered_record.timestamp)
        
# stream_to_img = {k:np.array(v) for k, v in stream_to_img.items()} 

# segmentation_mono = np.stack([[cv2.resize(i, (160, 120)) for i in stream_to_img['1201-1']], [cv2.resize(i, (160, 120)) for i in stream_to_img['1201-2']], [cv2.resize(i, (160, 120)) for i in stream_to_img['1201-3']], [cv2.resize(i, (160, 120)) for i in stream_to_img['1201-4']]], axis = 1)


# with open(f'data/mmt/real/quest2/raw/{key_id}.device_result.msgpack', 'rb') as f:
#         # Unpack the data from the file
#         data = msgpack.unpack(f)

# headsetTransformInWorld = np.array([data['frames'][i]['headsetTransform']['headsetTransformInWorld']['affine'] for i in range(len(data['frames']))])
# headsetTransformInWorld[:, :3, 3] /= 1000
# np.set_printoptions(precision=4, suppress=1)
# # plt.plot(headsetTransformInWorld[:, :3, 3])
# time_stamps_frames = [data['frames'][i]['arrivalTimestampInNs']/ 1000000000 for i in range(len(data['frames']))]
# start = np.abs(np.array(time_stamps_frames) - time_stamps[0]).argmin()
# end = (headsetTransformInWorld.shape[0] - segmentation_mono.shape[0]) - start


# import ipdb; ipdb.set_trace() 
# write_frames_to_video_io(stream_to_img['1201-4'], out_file_name=f"data/mmt/real/quest2/raw/{key_id}_1201-4.mp4", frame_rate=30)

# data_dump = {
#     "quest_images": stream_to_img,
#     "head_tracking": data,
#     "img_time_stamp": time_stamps
# }
# joblib.dump(data_dump, f"data/mmt/real/quest2/raw/{key_id}.pkl")


mujoco_joint_names = [
    'b_root', 
    'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_neck0', 'b_head', 
    'b_l_shoulder', 'p_l_scap', 'b_l_arm', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 
    'b_r_shoulder', 'p_r_scap', 'b_r_arm', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 
    'b_l_upleg', 'b_l_leg', "b_l_foot_twist", 
    'b_r_upleg', 'b_r_leg', "b_r_foot_twist", 
]

track_bodies = ['b_head']
# track_bodies = [ 'L_Knee', "L_Ankle", 'R_Knee', "R_Ankle", 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist',  'R_Shoulder', 'R_Elbow', 'R_Wrist']
track_idx = [mujoco_joint_names.index(i) for i in track_bodies]
standing_data = joblib.load("mmt/pkls/mmt_standing.pkl")

standing_key = '0_WIDC170.standing'
pose_quat_global_default = standing_data[standing_key]['pose_quat_global'][0:1].clone()
pose_quat_default = standing_data[standing_key]['pose_quat'][0:1].clone()
trans_orig_default = standing_data[standing_key]['trans_orig'][0:1].clone()
root_trans_offset_default = standing_data[standing_key]['root_trans_offset'][0:1].clone()
mmt_pose_params_default = standing_data[standing_key]['mmt_pose_params'][0:1].clone()
to_isaac_mat = sRot.from_euler('xyz', np.array([-np.pi/2, 0, 0]), degrees=False).as_matrix()


# head_up_rot = sRot.from_euler("xyz", [0, 0, np.pi/2])
head_up_rot = sRot.from_euler("xyz", [0, 0, -np.pi/2])
aria_transform_rot = sRot.from_euler("xyz", [np.pi / 2, 0, np.pi/2])
aria_transform_rot_dev = sRot.from_euler("xyz", [0, 0, np.pi])
# aria_transform_rot = sRot.from_matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

to_up_z = sRot.from_euler("xyz", [np.pi / 2, 0, 0])
# to_up_z = sRot.from_euler("xyz", [np.pi/2, 0, np.pi/2])
# to_up_z = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])

quest_data = {}
key_ids = [k.split("/")[-1].split(".")[0] for k in glob.glob("data/mmt/real/quest2/raw_processed/*.pkl")]

for key_id in key_ids:
    # if key_id != "capture00":
    #     continue
    quest_data_entry = joblib.load(f"data/mmt/real/quest2/raw_processed/{key_id}.pkl")
    quest_data[key_id] = quest_data_entry['test']

for k, v in tqdm(quest_data.items()):
    
    data_seq = {}  
    N = v['quest_sensor_data'].shape[0]
#     #### Aria head
    # head_rots =    head_up_rot * aria_transform_rot * sRot.from_matrix(v['quest_sensor_data'][:, :3, :3]) * aria_transform_rot.inv() * sRot.from_euler("xyz", [0, -np.pi/2, 0])
    head_rot_transfer = (head_up_rot * aria_transform_rot * sRot.from_matrix(v['quest_sensor_data'][:, :3, :3]) * aria_transform_rot.inv())
    
    head_rots = head_rot_transfer * sRot.from_quat([0.5, 0.5, 0.5, -0.5])
    rots_quat = head_rots.as_quat() 
    head_trans = v['quest_sensor_data'][:, :3, 3].copy() @ to_up_z.as_matrix().T
    
    height_diff = (head_trans[0, 2]- 1.5) 
    trans_diff =  head_trans[0, :2].copy()
    head_trans[:, 2] = head_trans[:, 2]  -  height_diff
    head_trans[:, :2] = head_trans[:, :2] - trans_diff

    poses_rot_t = np.concatenate([head_trans, rots_quat], axis = -1)[:, None]
    
    ####### Adjusti heading 
    heading_rot = np.zeros((N, 3))
    
    heading_rot[:, 2] = np.pi/2 + head_rot_transfer.as_euler("xyz")[:, 2]
    # heading_rot[:, 2] =  head_rot_transfer.as_euler("xyz")[:, 2]
    
    heading_rot = sRot.from_euler("xyz", heading_rot)
    root_rot_global = sRot.from_quat(pose_quat_global_default[:, 0].reshape(-1, 4))
    pose_quat_new = pose_quat_default.repeat(N, 1, 1).clone()
    pose_quat_new[:, 0] = torch.from_numpy((heading_rot * root_rot_global).as_quat())
    
    data_seq['pose_quat_global'] = pose_quat_global_default.repeat(N, 1, 1)
    data_seq['pose_quat'] = pose_quat_new

    data_seq['trans_orig'] = head_trans.copy()
    data_seq['root_trans_offset'] = torch.from_numpy(head_trans).clone()
    data_seq['root_trans_offset'][..., 2] = root_trans_offset_default[..., 2]
    # data_seq['trans_orig'][..., 2] = root_trans_offset_default[..., 2]
    
    data_seq['mmt_pose_params'] = mmt_pose_params_default.repeat(N, 1)
    data_seq['segmentation_mono'] = v["segmentation_mono"]
    
    data_seq['track_idx'] = track_idx
    
    
    
    data_seq.update({"quest_sensor_data": poses_rot_t, "fps": 30})
    
    joblib.dump({k: data_seq}, f"data/mmt/real/quest2/singles/11_9/{k}.pkl")
    
 
