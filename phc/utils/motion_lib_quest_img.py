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
from phc.utils.motion_lib_quest import MotionLibQuest
from phc.utils.motion_lib_base import compute_motion_dof_vels, DeviceCache, FixHeightMode
import cv2

from enum import Enum
from smpl_sim.utils.torch_ext import to_torch
JOINT_NAMES_PICK = [
       'b_root', 
       'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_neck0', 'b_head', 
       
       'b_l_shoulder', 'p_l_scap', 'b_l_arm', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 
       
       'b_r_shoulder', 'p_r_scap', 'b_r_arm', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 
        
       'b_l_upleg', 'b_l_leg', "b_l_foot_twist", 
       'b_r_upleg', 'b_r_leg', "b_r_foot_twist", 
]

from torchvision import transforms
class RandomGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img):
        gamma = self.gamma + np.random.uniform(-0.7, 0.7)
        return transforms.functional.adjust_gamma(img, gamma)

# Define your own GaussianNoise class
class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


def get_fisheye_mask(height = 120, width = 160):
    darken_factor = 1
    center_x, center_y = width // 2, height // 2
    max_radius = np.sqrt(center_x**2 + center_y**2)
    ys, xs = np.ogrid[:height, :width]
    distances_from_center = np.sqrt((xs - center_x)**2 + (ys - center_y)**2)
    threshold_radius = max_radius * (1 - darken_factor)

    scaling_factor = (distances_from_center - threshold_radius) / (max_radius - threshold_radius)

    scaling_factor = np.clip(scaling_factor, 0, 1)
    scaling_factor[scaling_factor < 0.85] = 0
    mask = (1 - scaling_factor**2)

    return mask

USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

def process_mono_images(images, augment = False):
    T, C, H, W = images.shape
    mask = get_fisheye_mask()
    if augment:
        images = images.view(T * C, 1, H, W)
        t_color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3)
        t_gamma = RandomGamma(1)
        t_noise = GaussianNoise(0, 10)
        
        img_transformed = torch.clamp(t_noise(images)/255, 0, 1)
        img_transformed = t_gamma(img_transformed)
        img_transformed = t_color_jitter(img_transformed) 
        img_transformed = img_transformed * mask
        img_transformed = img_transformed.view(T,  C, H, W)
    else:
        img_transformed = images/255
    
    return img_transformed
        
    
class MotionlibMode(Enum):
    file = 1
    directory = 2
    
class MotionLibQuestImg(MotionLibQuest):

    def __init__(self, motion_lib_cfg):
        # ZL Hack: can't really load .pkl for all the images. 
        super().__init__(motion_lib_cfg=motion_lib_cfg)
    
        self.load_mono = motion_lib_cfg.load_mono
        self.load_feat = motion_lib_cfg.load_feat
        self.load_heatmap = motion_lib_cfg.load_heatmap
        self.load_head_gt_3d = motion_lib_cfg.load_head_gt_3d
        
        joints_name_hp = copy.deepcopy(JOINT_NAMES_PICK)
        joints_name_hp.remove("b_head")
        joints_name_hp.remove("b_neck0")
        self.hp_indxes = [JOINT_NAMES_PICK.index(n) for n in joints_name_hp]
        
        return

    def load_motions(self, skeleton_trees, gender_betas, limb_weights, random_sample=True, start_idx=0, max_len=-1, augment_images = False):
        flags.augment_images = augment_images
        print(f"!!!!!!Augmenting images? {flags.augment_images}")
        print(f"!!!!!!Augmenting images? {flags.augment_images}")
        print(f"!!!!!!Augmenting images? {flags.augment_images}")
        motions = super().load_motions(skeleton_trees=skeleton_trees, gender_betas=gender_betas, limb_weights=limb_weights, random_sample=random_sample, start_idx=start_idx, max_len=max_len)
        
        if self.load_mono and "mono_images" in motions[0].__dict__: self.mono_images = torch.cat([m.mono_images for m in motions], dim=0).float()
        if self.load_heatmap and  "heatmaps" in motions[0].__dict__: self.heatmaps = torch.cat([m.heatmaps[:, :, self.hp_indxes] for m in motions], dim=0).float().to(self._device)
        if self.load_feat:  self.unreal_ego_feat = torch.cat([m.unreal_ego_feat  for m in motions], dim=0).float().to(self._device) # feat can load to features.
        if self.load_head_gt_3d and  "head_gt_3d" in motions[0].__dict__: self.head_gt_3d = torch.cat([m.head_gt_3d for m in motions], dim=0).float().to(self._device) # feat can load to features.
        return motions
    
    
        
    
    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, gender_betas,  mesh_parsers, config, queue, pid):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        max_len = config.max_length
        fix_height = config.fix_height
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
            pose_quat_global = curr_file['pose_quat_global'].clone()[start:end]
            B, J, N = pose_quat.shape
            
            # trans, trans_fix = MotionLibQuest.fix_trans_height(curr_file['mmt_pose_params'], trans, curr_gender_beta, mesh_parsers, fix_height_mode = fix_height) # No mesh loader for Quest (momentum) humanoid
            #####################

            sk_state = SkeletonState.from_rotation_and_root_translation(copy.deepcopy(skeleton_trees[f]), pose_quat, trans, is_local=True)

            curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
            
            curr_dof_vels = compute_motion_dof_vels(curr_motion)

            #################################### Loading images ####################################
            
            if "segmentation_mono" in curr_file: curr_motion.mono_images = process_mono_images(to_torch(curr_file['segmentation_mono'][start:end]), augment=flags.augment_images)
            if 'heatmaps' in curr_file:  curr_motion.heatmaps = to_torch(curr_file['heatmaps'][start:end])
            if 'unreal_ego_feat' in curr_file:  curr_motion.unreal_ego_feat = curr_file['unreal_ego_feat'][start:end]
            # if 'head_gt_3d' in curr_file:  curr_motion.head_gt_3d = to_torch(curr_file['head_gt_3d'])[start:end]
            if 'head_gt_3d' in curr_file:  curr_motion.head_gt_3d = to_torch(curr_file['head_fk_gt_3d'])[start:end]
            
            # shape = curr_motion.mono_images.shape # TO make code run. 
            # curr_motion.mono_images = torch.cat([curr_motion.mono_images, torch.zeros((shape[0], shape[1], 8, shape[3]))], dim = -2)
            
            
            
            # RT = curr_file['cam_right']['T'][start:end]
            # RR = curr_file['cam_right']['R'][start:end]
            # RD = curr_file['cam_right']['distortion']
            # RK = curr_file['cam_right']['intrinsics']
            
            # R_transform_quat_ext = torch.from_numpy(sRot.from_matrix(RR).as_quat())[:, None]
            # R_pose_quat_global = torch_utils.quat_mul(R_transform_quat_ext.repeat(1, pose_quat_global.shape[1], 1), pose_quat_global)
            # R_trans = trans + RT
            # R_sk_state = SkeletonState.from_rotation_and_root_translation(copy.deepcopy(skeleton_trees[f]), R_pose_quat_global, R_trans, is_local=False)
            # R_motion = SkeletonMotion.from_skeleton_state(R_sk_state, curr_file.get("fps", 30))
            # R_curr_dof_vels = compute_motion_dof_vels(R_motion)
            # R_motion.dof_vels = curr_dof_vels
            # R_motion.gender_beta = curr_gender_beta
            # R_motion.mono_images = mono_images 
            # curr_motion.R_motion = R_motion
            
            #################################### Loading images ####################################
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