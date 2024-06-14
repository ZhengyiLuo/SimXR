import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from ast import If
import numpy as np
import yaml
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
from phc.utils.motion_lib_smpl import MotionLibSMPL
from phc.utils.motion_lib_base import compute_motion_dof_vels, DeviceCache, FixHeightMode
import cv2
from enum import Enum
from smpl_sim.utils.torch_ext import to_torch

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


    
class MotionLibSMPLImg(MotionLibSMPL):
    
    
    def __init__(self, motion_lib_cfg):
        super().__init__(motion_lib_cfg = motion_lib_cfg)
        data_dir = "data/smpl"
        self.load_mono = motion_lib_cfg.load_mono
        self.load_feat = motion_lib_cfg.load_feat
        self.load_heatmap = motion_lib_cfg.load_heatmap
        self.load_head_gt_3d = motion_lib_cfg.load_head_gt_3d
        
        return
    

    def load_motions(self, skeleton_trees, gender_betas, limb_weights, random_sample=True, start_idx=0, max_len=-1, augment_images = False):
        motions = super().load_motions(skeleton_trees=skeleton_trees, gender_betas=gender_betas, limb_weights=limb_weights, random_sample=random_sample, start_idx=start_idx, max_len=max_len)
        
        if self.load_mono and "mono_images" in motions[0].__dict__: self.mono_images = torch.cat([to_torch(m.mono_images)   for m in motions], dim=0).float().to(self._device)
        if self.load_heatmap and  "heatmaps" in motions[0].__dict__: self.heatmaps = torch.cat([to_torch(m.heatmaps) for m in motions], dim=0).float().to(self._device)
        if self.load_feat:  self.unreal_ego_feat = torch.cat([to_torch(m.unreal_ego_feat)   for m in motions], dim=0).float().to(self._device) # feat can load to features.
        if self.load_head_gt_3d and  "head_gt_3d" in motions[0].__dict__: self.head_gt_3d = torch.cat([to_torch(m.head_gt_3d) for m in motions], dim=0).float().to(self._device) # feat can load to features.

        return motions
    
    
    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, shape_params, mesh_parsers, config, queue, pid):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        max_len = config.max_length
        fix_height = config.fix_height
        res = {}
        for f in range(len(motion_data_list)):
            assert (len(ids) == len(motion_data_list))
            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]
            
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]
            curr_gender_beta = shape_params[f]
            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = curr_file['root_trans_offset'].clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()
            pose_quat_global = curr_file['pose_quat_global'][start:end].copy()

            B, J, N = pose_quat_global.shape
            
            
            if (not flags.im_eval) and (not flags.test) and (not flags.real_traj):
                # if True:
                random_rot = np.zeros(3)
                random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
                random_heading_rot = sRot.from_euler("xyz", random_rot)
                pose_aa[:, :3] = torch.tensor((random_heading_rot * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec())
                pose_quat_global = (random_heading_rot * sRot.from_quat(pose_quat_global.reshape(-1, 4))).as_quat().reshape(B, J, N)
                trans = torch.matmul(trans.float(), torch.from_numpy(random_heading_rot.as_matrix().T).float())
            trans, trans_fix = MotionLibSMPL.fix_trans_height(pose_aa, trans, curr_gender_beta, mesh_parsers, fix_height_mode = fix_height)
            
            pose_quat_global = to_torch(pose_quat_global)
            sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees[f], pose_quat_global, trans, is_local=False)

            curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
            curr_dof_vels = compute_motion_dof_vels(curr_motion)
            #################################### Loading images ####################################
            if "segmentation_mono" in curr_file: curr_motion.mono_images = curr_file['segmentation_mono'][start:end]/255
            if 'heatmaps' in curr_file:  curr_motion.heatmaps = curr_file['heatmaps'][start:end]
            if 'unreal_ego_feat' in curr_file:  curr_motion.unreal_ego_feat = curr_file['unreal_ego_feat'][start:end]
            if 'head_gt_3d' in curr_file:  curr_motion.head_gt_3d = curr_file['head_gt_3d'][start:end]
            
            #################################### Loading images ####################################
            if flags.real_traj:
                quest_sensor_data = to_torch(curr_file['quest_sensor_data'])[start:end].clone()
                quest_trans = quest_sensor_data[..., :3]
                quest_rot = quest_sensor_data[..., 3:]
                
                # (curr_motion.global_translation[:, skeleton_trees[f].node_names.index("Head"), :3]  - quest_trans[:, 4, :3]).norm(dim=-1).mean()
                
                trans_fix_real = quest_trans[0, 0, -1] - curr_motion.global_translation[0, skeleton_trees[f].node_names.index("Head"), -1] 
                quest_trans[:, 0, -1] -= trans_fix_real # Fix trans
                    
                
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
        
    def get_images(self, motion_ids, motion_times, offset=None):
        return_dict = {}
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]
        
        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)
        
        ############################## Images ##############################
        blend_idx = blend
        blend_idx = (blend_idx >= 0.5).long() # Can not interperlate. 
        f0 = torch.gather(torch.stack([f0l, f1l], dim = -1), 1, blend_idx).squeeze(1)
        
        if self.load_mono and "mono_images" in self.__dict__:
            mono_images = self.mono_images[f0.cpu()]
            return_dict['mono_images'] = mono_images.to(self._device)
        
        if self.load_heatmap and "heatmaps" in self.__dict__:
            return_dict['heatmaps'] = self.heatmaps[f0]
            
        if self.load_feat and "unreal_ego_feat" in self.__dict__:
            return_dict['img_feats'] = self.unreal_ego_feat[f0]
            
        if self.load_head_gt_3d and "head_gt_3d" in self.__dict__:
            return_dict['head_gt_3d'] = self.head_gt_3d[f0]
            
        return return_dict    
    
    def get_motion_state(self, motion_ids, motion_times, offset=None, load_img = False):
        return_dict = {}
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        
        if load_img:
            ############################## Images ##############################
            blend_idx = blend
            blend_idx = (blend_idx >= 0.5).long() # Can not interperlate. 
            f0 = torch.gather(torch.stack([f0l, f1l], dim = -1), 1, blend_idx).squeeze(1)
            
            if self.load_mono and "mono_images" in self.__dict__:
                mono_images = self.mono_images[f0.cpu()]
                return_dict['mono_images'] = mono_images.to(self._device)
            
            if self.load_heatmap and "heatmaps" in self.__dict__:
                return_dict['heatmaps'] = self.heatmaps[f0]
                
            if self.load_feat and "unreal_ego_feat" in self.__dict__:
                return_dict['img_feats'] = self.unreal_ego_feat[f0]
                
            if self.load_feat and "head_gt_3d" in self.__dict__:
                return_dict['head_gt_3d'] = self.head_gt_3d[f0]
        else:
            mono_images = None
            
        if flags.real_traj:
            q_body_ang_vel0, q_body_ang_vel1 = self.q_gavs[f0l], self.q_gavs[f1l]
            q_rb_rot0, q_rb_rot1 = self.q_grs[f0l], self.q_grs[f1l]
            q_rg_pos0, q_rg_pos1 = self.q_gts[f0l, :], self.q_gts[f1l, :]
            q_body_vel0, q_body_vel1 = self.q_gvs[f0l], self.q_gvs[f1l]

            q_ang_vel = (1.0 - blend_exp) * q_body_ang_vel0 + blend_exp * q_body_ang_vel1
            q_rb_rot = torch_utils.slerp(q_rb_rot0, q_rb_rot1, blend_exp)
            q_body_vel = (1.0 - blend_exp) * q_body_vel0 + blend_exp * q_body_vel1
            if offset is None:
                q_rg_pos = (1.0 - blend_exp) * q_rg_pos0 + blend_exp * q_rg_pos1
            else:
                q_rg_pos = (1.0 - blend_exp) * q_rg_pos0 + blend_exp * q_rg_pos1 + offset[..., None, :]  # ZL: apply offset
            
            rg_pos[:, self.track_idx] = q_rg_pos
            rb_rot[:, self.track_idx] = q_rb_rot
            body_vel[:, self.track_idx] = q_body_vel
            body_ang_vel[:, self.track_idx] = q_ang_vel
        

        return_dict.update({
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "local_rot": local_rot.clone(), 
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "motion_limb_weights": self._motion_limb_weights[motion_ids],
        })
        
        return return_dict