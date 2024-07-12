from ast import If
import numpy as np
import os
import yaml
import os.path as osp
from tqdm import tqdm

from isaacgym.torch_utils import *
from phc.utils import torch_utils
import joblib
import torch
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import torch.multiprocessing as mp
import copy
import gc
from phc.utils.flags import flags
import random
from scipy.spatial.transform import Rotation as sRot
from phc.utils.motion_lib_base import MotionLibBase, DeviceCache, compute_motion_dof_vels, FixHeightMode


USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy

    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


class MotionLibQuest(MotionLibBase):

    def __init__(self, motion_file, device, fix_height=FixHeightMode.full_fix, masterfoot_conifg=None, min_length=-1, im_eval=False, multi_thread=True):
        super().__init__(motion_file=motion_file, device=device, fix_height=fix_height, masterfoot_conifg=masterfoot_conifg, min_length=min_length, im_eval=im_eval, multi_thread=multi_thread)
        return
    
        
    @staticmethod
    def fix_trans_height(mmt_params, trans, curr_scales, mesh_parsers, fix_height_mode = FixHeightMode.full_fix):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        
        with torch.no_grad():
            height_tolorance = 0.0
            frame_check = 30
            verts, model_params, joint_params, state_t, state_r, state_s = mesh_parsers(mmt_params[:frame_check, :], curr_scales, mesh_parsers.mesh_vertices) # only uses the first 60 frames, which usually is a calibration phase. To conserve memory. 
            verts = verts / 100
            
            if fix_height_mode == FixHeightMode.full_fix or fix_height_mode == FixHeightMode.ankle_fix: # MMT ankle fix not implemented yet
                diff_fix = verts[:, :, 1].min() # ZL: fix the height to be 0
                # import ipdb; ipdb.set_trace()
                trans[..., -1] -= diff_fix 
            return trans, diff_fix
        

    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, gender_betas, fix_height, mesh_parsers, masterfoot_config, max_len, queue, pid):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        
        res = {}
        for f in range(len(motion_data_list)):
            assert (len(ids) == len(motion_data_list))
            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = ".".join(motion_data_list[f].split("/")[-1].split(".")[:-1])
                curr_file = joblib.load(curr_file)[key]
                
            curr_gender_beta = gender_betas[f]
            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = curr_file['root_trans_offset'].clone()[start:end]
            pose_quat = curr_file['pose_quat'].clone()[start:end]
            B, J, N = pose_quat.shape

            ###### ZL: randomize the heading ######
            # random_rot = np.zeros(3)
            # random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
            # random_heading_rot = sRot.from_euler("xyz", random_rot)
            # pose_quat[:, 0] = torch.tensor((random_heading_rot * sRot.from_quat(pose_quat[:, 0].reshape(-1, 4))).as_quat())
            # trans = torch.matmul(trans, torch.from_numpy(random_heading_rot.as_matrix().T).float())
            ###### ZL: randomize the heading ######
            
            trans, trans_fix = MotionLibMMT.fix_trans_height(curr_file['mmt_pose_params'], trans, curr_gender_beta, mesh_parsers, fix_height_mode = fix_height)

            sk_state = SkeletonState.from_rotation_and_root_translation(copy.deepcopy(skeleton_trees[f]), pose_quat, trans, is_local=True)

            curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
            curr_dof_vels = compute_motion_dof_vels(curr_motion)
            if flags.real_traj:
                
                quest_sensor_data = to_torch(curr_file['quest_sensor_data'])[start:end]
                quest_trans = quest_sensor_data[..., :3]
                quest_rot = quest_sensor_data[..., 3:]
                
                ##### Has to account for the body shap difference here. 
                # verts, model_params, joint_params, state_t, state_r, state_s  = mesh_parsers(curr_file['mmt_pose_params'][:60, :], curr_file['scale'][:60], mesh_parsers.mesh_vertices)
                # verts_mean, _, _, state_t_mean, _, _  = mesh_parsers(curr_file['mmt_pose_params'][:60, :], curr_gender_beta, mesh_parsers.mesh_vertices)
                # 
                
                ### Equivalent, just use the current head height. 
                # import ipdb; ipdb.set_trace()
                # print("hacking height")
                # print("hacking height")
                # print("hacking height")
                # trans_fix_real = quest_trans[0, 0, -1] - curr_motion.global_translation[0, skeleton_trees[f].node_names.index("b_head"), -1]  + 0.03
                # quest_trans[:, 0, -1] -= trans_fix_real # Fix trans
                # ##### Has to account for the body shap difference here.
                
                global_angular_vel = SkeletonMotion._compute_angular_velocity(quest_rot, time_delta=1 / curr_file['fps'])
                linear_vel = SkeletonMotion._compute_velocity(quest_trans, time_delta=1 / curr_file['fps'])
                quest_motion = {"global_angular_vel": global_angular_vel, "linear_vel": linear_vel, "quest_trans": quest_trans, "quest_rot": quest_rot}
                curr_motion.quest_motion = quest_motion

            curr_motion.dof_vels = curr_dof_vels
            curr_motion.gender_beta = curr_gender_beta
            res[curr_id] = (curr_file, curr_motion)

        if not queue is None:
            queue.put(res)
        else:
            return res
