from isaacgym import gymapi
from isaacgym import gymtorch

import os.path as osp
from typing import OrderedDict
import torch
import numpy as np
import phc.env.tasks.humanoid_im as humanoid_im
from phc.utils.motion_lib_base import local_rotation_to_dof_vel
from phc.utils.motion_lib_quest_img import  MotionLibQuestImg
from phc.utils.motion_lib_smpl_img import  MotionLibSMPLImg
from phc.env.tasks.humanoid_amp import remove_base_rot

from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
from phc.utils import torch_utils
import torch.nn as nn
from easydict import EasyDict

import phc.utils.pytorch3d_transforms as ptr
from phc.utils.torch_utils import *
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import cv2
from collections import deque
import imageio
from tqdm import tqdm
from datetime import datetime
from phc.learning.network_loader import load_z_encoder, load_z_decoder, load_mcp_mlp, load_pnn
from phc.utils.motion_lib_base import FixHeightMode
import copy

class HumanoidEgoImg(humanoid_im.HumanoidIm):
    
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.img_dim = cfg['env'].get("img_dim", [4, 128, 160])
        self.img_aug = cfg['env'].get("img_aug", False)
        self.heat_dim = cfg['env'].get("heat_dim", [2, 11, 40, 30])
        self.img_latent_dim = cfg['env'].get("img_latent_dim", 512)
        self.distill_z_model = cfg['env'].get("distill_z_model", False)
        self.use_unrealego = cfg['env'].get("use_unrealego", False)
        self.use_visibility_branch = cfg['env'].get("use_visibility_branch", False)
        self.use_convnext = cfg['env'].get("use_convnext", False)
        self.use_resnet_no_bn = cfg['env'].get("use_resnet_no_bn", False)
        self.use_resnet_gn = cfg['env'].get("use_resnet_gn", False)
        self.use_siamese = cfg['env'].get("use_siamese", False)
        self.pretrain = cfg['env'].get("pretrain", False)
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        
        if self.distill and not flags.test:
            check_points = [torch_ext.load_checkpoint(ck_path) for ck_path in self.models_path]
            self.distill_model_config = self.cfg['env']['distill_model_config']
            self.z_activation = self.distill_model_config['z_activation']
            
            ### Loading Distill Model ###
            self.embedding_size_distill = self.distill_model_config.get("embedding_size", 1024)
            self.embedding_norm_distill = self.distill_model_config.get("embedding_norm", -1)
            self.fut_tracks_distill = self.distill_model_config.get("fut_tracks", False)
            self.num_traj_samples_distill = self.distill_model_config.get("numTrajSamples", -1)
            self.traj_sample_timestep_distill = self.distill_model_config.get("trajSampleTimestepInv", -1)
            self.fut_tracks_dropout_distill = self.distill_model_config.get('fut_tracks_dropout', False)
            self.z_all_distill = self.distill_model_config.get('z_all', False)
            self.root_height_obs_distill = self.distill_model_config.get('root_height_obs', True)
            self.z_type_distill = self.distill_model_config.get("z_type", "sphere")
            self.obs_v_distill = self.distill_model_config.get("obs_v", 6)
            self.track_bodies_distill = self.distill_model_config.get("trackBodies", None)
            
            if not self.track_bodies_distill is None:
                self.track_bodies_id_distill =self._build_key_body_ids_tensor(self.track_bodies_distill)
            else:
                self.track_bodies_id_distill = self._track_bodies_id
                
            if self.distill_z_model:
                ### Loading Distill Model ###
                self.decoder = load_z_decoder(check_points[0], activation = self.z_activation, z_type = self.z_type_distill, device = self.device) 
                self.encoder = load_z_encoder(check_points[0], activation = self.z_activation, z_type = self.z_type_distill, device = self.device)
            else:
                self.has_pnn_distill = self.distill_model_config.get("has_pnn", False)
                self.has_lateral_distill = self.distill_model_config.get("has_lateral", False)
                self.num_prim_distill = self.distill_model_config.get("num_prim", 3)
                self.discrete_moe_distill = self.distill_model_config.get("discrete_moe", False)
                
                if self.has_pnn_distill:
                    # assert (len(self.models_path) == 2)
                    self.pnn = load_pnn(check_points[0], num_prim = self.num_prim_distill, has_lateral = self.has_lateral_distill, activation = self.z_activation, device = self.device)
                    self.running_mean, self.running_var = check_points[0]['running_mean_std']['running_mean'], check_points[0]['running_mean_std']['running_var']
                else:
                    self.encoder = load_mcp_mlp(check_points[0], activation = self.z_activation, device = self.device)
                    
            self.running_mean, self.running_var = check_points[-1]['running_mean_std']['running_mean'], check_points[-1]['running_mean_std']['running_var']
            
        self.ref_mono_images = None
            
        if self.save_kin_info:
            self.kin_dict = OrderedDict()
            self.kin_dict.update({
                "gt_action": torch.zeros([self.num_envs, self._num_actions]),
                }) # current root pos + root for future aggergration
            if self.use_unrealego or self.use_visibility_branch:
                self.kin_dict['heatmaps'] = torch.zeros([self.num_envs, *self.heat_dim]).to(self.device)
                self.ref_heatmap = torch.zeros([self.num_envs, *self.heat_dim]).to(self.device)
        

    
    def create_o3d_viewer(self):
        super().create_o3d_viewer()
        self._video_queue_o3d_img = deque(maxlen=self.max_video_queue_size)
        self._video_path_o3d_img = osp.join("output", "renderings", f"{self.cfg_name}-%s-o3d_mono.mp4")
        self.recording_state_change_o3d_img = False
    
    def get_running_mean_size(self):
        
        if self.partial_running_mean:
            if self.pretrain:
                return (self.get_obs_size(), )
            else:
                return (self.get_obs_size() - np.prod(self.img_dim), )
        else:
            return (self.get_obs_size(), )
        

    def render(self, sync_frame_time = False):
        super().render(sync_frame_time=sync_frame_time)
        if ((not self.headless) or self.viewer_o3d) and not self.ref_mono_images is None:
            
            ref_image = self.ref_mono_images.squeeze().numpy()
            if len(ref_image) == 4:
                mono_image = np.concatenate([np.concatenate(ref_image[[1, 2]], axis = 1), np.concatenate(ref_image[[0, 3]], axis = 1)], axis = 0)
            else:
                mono_image = np.concatenate(self.ref_mono_images.squeeze().numpy(), axis = 1)
            
            
            mono_image_x2 = cv2.resize(mono_image, (mono_image.shape[1]* 2, mono_image.shape[0] * 2))
            
            if self.control_i == 0:
                if self.recording_state_change_o3d_img:
                    if not self.recording:
                        self.writer_o3d_img.close()
                        del self.writer_o3d_img
                        
                        print(f"============ Video finished writing O3D Img {self.curr_video_o3d_img_file_name}============")
                    else:
                        print(f"============ Writing video O3D Img ============")
                        
                    self.recording_state_change_o3d_img = False
                
                if self.recording:
                    if not "writer_o3d_img" in self.__dict__:
                        print(f"============ Writing video O3D ============")
                        curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                        self.curr_video_o3d_img_file_name = self._video_path_o3d_img % curr_date_time
                        self.writer_o3d_img = imageio.get_writer(self.curr_video_o3d_img_file_name, fps=30, macro_block_size=None)
                    self.writer_o3d_img.append_data((mono_image * 255).astype(np.uint8))
            
            if self.num_envs == 1:
                cv2.imshow("mono image", mono_image_x2)
                cv2.waitKey(1)
                
        
    def _load_motion(self, motion_train_file, motion_test_file=[]):
        assert (self._dof_offsets[-1] == self.num_dof)
        load_feat = False
        load_heatmap = self.use_unrealego or self.use_visibility_branch
        load_mono = len(self.img_dim) > 1 or not self.headless
        
        if self.humanoid_type in ["quest"]:
            self._motion_lib = MotionLibQuestImg(motion_file=motion_train_file,  device=self.device, min_length=self._min_motion_len, fix_height= self.height_fix_mode, load_feat=load_feat, load_mono = load_mono, load_heatmap=load_heatmap) # Use ankle fix for image-based data
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=not flags.test, max_len=-1 if flags.test else self.max_len, augment_images=self.img_aug and not flags.real_traj and not flags.test)
            self._motion_train_lib = self._motion_eval_lib = self._motion_lib
        elif self.humanoid_type in ["smpl", "smplh", "smplx"]:
            motion_lib_cfg = EasyDict({
                "motion_file": motion_train_file,
                "device": torch.device("cpu"),
                "fix_height": FixHeightMode.full_fix,
                "min_length": self._min_motion_len,
                "max_length": self.max_len,
                "im_eval": flags.im_eval,
                "multi_thread": True ,
                "smpl_type": self.humanoid_type,
                "randomrize_heading": True,
                "device": self.device,
                "load_feat": load_feat, 
                "load_heatmap": load_heatmap, 
                "load_mono": load_mono, 
                "load_head_gt_3d": False,
            })
            self._motion_lib = MotionLibSMPLImg(motion_lib_cfg) # Use ankle fix for image-based data
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=not flags.test, max_len=-1 if flags.test else self.max_len)
            self._motion_train_lib = self._motion_eval_lib = self._motion_lib
            
        return
    
    def resample_motions(self):

        print("Partial solution, only resample motions...")
        # if self.hard_negative:
            # self._motion_lib.update_sampling_weight()

        if flags.test:
            self.forward_motion_samples()
        else:
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, limb_weights=self.humanoid_limb_and_weights.cpu(), gender_betas=self.humanoid_shapes.cpu(), random_sample=(not flags.test) and (not self.seq_motions),
                                          max_len=-1 if flags.test else self.max_len, augment_images=self.img_aug and not flags.real_traj and not flags.test)  # For now, only need to sample motions since there are only 400 hmanoids
            print(f"Augmenting images? {self.img_aug and not flags.real_traj}")

            # self.reset() #
            # print("Reasmpling and resett!!!.")

            time = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset
            root_res = self._motion_lib.get_root_pos_smpl(self._sampled_motion_ids, time)
            self._global_offset[:, :2] = self._humanoid_root_states[:, :2] - root_res['root_pos'][:, :2]
            self.reset()
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            if self.obs_v == 1:
                obs_size = len(self._track_bodies) * self._num_traj_samples * 15
            elif self.obs_v == 2:  # + dofdiff
                obs_size = len(self._track_bodies) * self._num_traj_samples * 15
                obs_size += (len(self._track_bodies) - 1) * self._num_traj_samples * 3
            elif self.obs_v == 3:  # reduced number
                obs_size = len(self._track_bodies) * self._num_traj_samples * 9
            elif self.obs_v == 4:  # 10 steps
                obs_size = len(self._track_bodies) * self._num_traj_samples * 15 * 5
            elif self.obs_v == 5:  # one hot vector for type of motions
                obs_size = len(self._track_bodies) * self._num_traj_samples * 15 + 4
            elif self.obs_v == 6:  # local+ dof + pos (not diff)
                obs_size = len(self._track_bodies) * self._num_traj_samples * 24
                if not self.pretrain:
                    obs_size += np.prod(self.img_dim)

            elif self.obs_v == 7:  # local+ dof + pos (not diff)
                obs_size = len(self._track_bodies) * self._num_traj_samples * 9  # linear position + velocity
                obs_size += np.prod(self.img_dim)

        return obs_size
    
    def get_task_obs_size_detail(self):
        task_obs_detail = OrderedDict()
        task_obs_detail['target'] = self.get_task_obs_size()
        task_obs_detail['fut_tracks'] = self._fut_tracks
        task_obs_detail['num_traj_samples'] = self._num_traj_samples
        task_obs_detail['obs_v'] = self.obs_v
        task_obs_detail['track_bodies'] = self._track_bodies
        task_obs_detail['models_path'] = self.models_path

        # Dev
        task_obs_detail['num_prim'] = self.cfg['env'].get("num_prim", 2)
        task_obs_detail['training_prim'] = self.cfg['env'].get("training_prim", 1)
        task_obs_detail['actors_to_load'] = self.cfg['env'].get("actors_to_load", 2)
        task_obs_detail['has_lateral'] = self.cfg['env'].get("has_lateral", True)

        ### For Z
        task_obs_detail['img_size'] = self.img_dim
        task_obs_detail['img_latent_dim'] = self.img_latent_dim
        task_obs_detail['use_unrealego'] = self.use_unrealego
        task_obs_detail['use_visibility_branch'] = self.use_visibility_branch
        task_obs_detail['use_convnext'] = self.use_convnext
        task_obs_detail['use_resnet_no_bn'] = self.use_resnet_no_bn
        task_obs_detail['use_resnet_gn'] = self.use_resnet_gn
        task_obs_detail['use_siamese'] = self.use_siamese
        task_obs_detail['pretrain'] = self.pretrain
        if self.obs_v == 6:
            task_obs_detail['target_size'] = len(self._track_bodies) * self._num_traj_samples * (24)
        elif self.obs_v == 7:
            task_obs_detail['target_size'] = len(self._track_bodies) * self._num_traj_samples * (9)
            
        return task_obs_detail
    
    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None, load_img = False):
        ## Ading load_img flag. 
        
        if offset is None  or not "motion_ids" in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() + (self.ref_motion_cache['offset'] - offset).abs().sum() > 0 :
            self.ref_motion_cache['motion_ids'] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['motion_times'] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['offset'] = offset.clone() if not offset is None else None
        else:
            return self.ref_motion_cache
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset, load_img = load_img)
        del self.ref_motion_cache
        self.ref_motion_cache = motion_res

        return self.ref_motion_cache

    
    def _compute_task_obs(self, env_ids=None, save_buffer = True, return_img = True):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]

        curr_gender_betas = self.humanoid_shapes[env_ids]

        if self._fut_tracks:
            time_steps = self._num_traj_samples
            B = env_ids.shape[0]
            time_internals = torch.arange(time_steps).to(self.device).repeat(B).view(-1, time_steps) * self._traj_sample_timestep
            motion_times_steps = ((self.progress_buf[env_ids, None] + 1) * self.dt + time_internals + self._motion_start_times[env_ids, None] + self._motion_start_times_offset[env_ids, None]).flatten()  # Next frame, so +1
            env_ids_steps = env_ids.repeat_interleave(time_steps)
            motion_res = self._get_state_from_motionlib_cache(env_ids_steps, motion_times_steps, self._global_offset[env_ids].repeat_interleave(time_steps, dim=0).view(-1, 3), load_img = True)  # pass in the env_ids such that the motion is in synced.

        else:
            motion_times = (self.progress_buf[env_ids] + 1) * self.dt + self._motion_start_times[env_ids] + self._motion_start_times_offset[env_ids]  # Next frame, so +1
            time_steps = 1
            motion_res = self._get_state_from_motionlib_cache(env_ids, motion_times, self._global_offset[env_ids], load_img = True)  # pass in the env_ids such that the motion is in synced.

        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel= \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        
        ref_mono_images = motion_res["mono_images"] if "mono_images" in motion_res else None
        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]

        body_pos_subset = body_pos[..., self._track_bodies_id, :]
        body_rot_subset = body_rot[..., self._track_bodies_id, :]
        body_vel_subset = body_vel[..., self._track_bodies_id, :]
        body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]

        ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
        ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
        ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
        ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]

        if self.obs_v != 6 and self.obs_v != 7:
            raise NotImplementedError
        elif self.obs_v == 6:
            if self.zero_out_far:
                close_distance = self.close_distance
                distance = torch.norm(root_pos - ref_rb_pos_subset[..., 0, :], dim=-1)

                zeros_subset = distance > close_distance
                ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
                ref_rb_rot_subset[zeros_subset, 1:] = body_rot_subset[zeros_subset, 1:]
                ref_body_vel_subset[zeros_subset, :] = body_vel_subset[zeros_subset, :]
                ref_body_ang_vel_subset[zeros_subset, :] = body_ang_vel_subset[zeros_subset, :]
                self._point_goal[env_ids] = distance

                far_distance = self.far_distance  # does not seem to need this in particular...
                vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                ref_rb_pos_subset[vector_zero_subset, 0] = ((ref_rb_pos_subset[vector_zero_subset, 0] - body_pos_subset[vector_zero_subset, 0]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 0]

            if self._occl_training:
                # ranomly occlude some of the body parts
                random_occlu_idx = self.random_occlu_idx[env_ids]
                ref_rb_pos_subset[random_occlu_idx] = body_pos_subset[random_occlu_idx]
                ref_rb_rot_subset[random_occlu_idx] = body_rot_subset[random_occlu_idx]
                ref_body_vel_subset[random_occlu_idx] = body_vel_subset[random_occlu_idx]
                ref_body_ang_vel_subset[random_occlu_idx] = body_ang_vel_subset[random_occlu_idx]

            obs = humanoid_im.compute_imitation_observations_v6(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, time_steps, self._has_upright_start)
            
            if return_img:
                if len(self.img_dim) > 1:
                    if not self.pretrain:
                        obs = torch.cat([obs, motion_res["mono_images"].view(obs.shape[0], -1)], dim = -1)
                        if self.use_unrealego :
                            self.ref_heatmap[env_ids] = motion_res["heatmaps"]
                        elif self.use_visibility_branch:
                            heatmaps = motion_res["heatmaps"]
                            B, V, J, H, W = heatmaps.shape
                            self.ref_heatmap[env_ids] = (heatmaps.reshape(B, V, J, H * W).max(-1)[0] > 0.95).float()
                else:
                    if "img_feats" in motion_res:
                        obs = torch.cat([obs, motion_res["img_feats"].view(obs.shape[0], -1)], dim = -1)
                    else:
                        obs = torch.cat([obs, torch.zeros((obs.shape[0], self.img_dim[0])).to(obs)], dim = -1)
                
        elif self.obs_v == 7:

            if self.zero_out_far:
                close_distance = self.close_distance
                distance = torch.norm(root_pos - ref_rb_pos_subset[..., 0, :], dim=-1)

                zeros_subset = distance > close_distance
                ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
                ref_body_vel_subset[zeros_subset, :] = body_vel_subset[zeros_subset, :]
                self._point_goal[env_ids] = distance

                far_distance = self.far_distance  # does not seem to need this in particular...
                vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                ref_rb_pos_subset[vector_zero_subset, 0] = ((ref_rb_pos_subset[vector_zero_subset, 0] - body_pos_subset[vector_zero_subset, 0]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 0]

            if self._occl_training:
                # ranomly occlude some of the body parts
                random_occlu_idx = self.random_occlu_idx[env_ids]
                ref_rb_pos_subset[random_occlu_idx] = body_pos_subset[random_occlu_idx]
                ref_rb_rot_subset[random_occlu_idx] = body_rot_subset[random_occlu_idx]

            obs = humanoid_im.compute_imitation_observations_v7(root_pos, root_rot, body_pos_subset, body_vel_subset, ref_rb_pos_subset, ref_body_vel_subset, time_steps, self._has_upright_start)
        if save_buffer:
            self.ref_body_pos[env_ids] = ref_rb_pos
            self.ref_body_vel[env_ids] = ref_body_vel
            self.ref_body_rot[env_ids] = ref_rb_rot
            self.ref_body_pos_subset[env_ids] = ref_rb_pos_subset
            self.ref_dof_pos[env_ids] = ref_dof_pos
            self.ref_mono_images = ref_mono_images
                
        del motion_res
        return obs


     
    def step(self, actions):
        
        # if self.distill:
        if self.distill and not flags.test and self.save_kin_info:
            with torch.no_grad():
            # Apply trained Model.

            ################ GT-Action ################
                temp_tracks = self._track_bodies_id
                if self.track_bodies_distill is not None:
                    self._track_bodies_id = self.track_bodies_id_distill
                else:
                    self._track_bodies_id = self._full_track_bodies_id
                temp_fut, temp_fut_drop, temp_timestep, temp_num_steps, temp_root_height_obs, temp_obs_v = self._fut_tracks, self._fut_tracks_dropout, self._traj_sample_timestep, self._num_traj_samples, self._root_height_obs, self.obs_v
                
                self._fut_tracks, self._fut_tracks_dropout, self._traj_sample_timestep, self._num_traj_samples, self._root_height_obs, self.obs_v = self.fut_tracks_distill, self.fut_tracks_dropout_distill, 1/self.traj_sample_timestep_distill, self.num_traj_samples_distill, self.root_height_obs_distill, self.obs_v_distill
                if self.root_height_obs_distill != temp_root_height_obs:
                    self_obs = self.obs_buf[:, :self.get_self_obs_size()]
                    self_obs = torch.cat([self._rigid_body_pos[:, 0, 2:3], self_obs], dim = -1) # Add root Height Obs
                    # self_obs = self._compute_humanoid_obs() # torch.cat([self._rigid_body_pos[:, 0, 2:3], self_obs], dim = -1) - self._compute_humanoid_obs()
                    
                    self_obs_size = self_obs.shape[-1]
                    self_obs = ((self_obs - self.running_mean.float()[:self_obs_size]) / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05))
                else:
                    self_obs_size = self.get_self_obs_size()
                    self_obs = ((self.obs_buf[:, :self_obs_size] - self.running_mean.float()[:self_obs_size]) / torch.sqrt(self.running_var.float()[:self_obs_size] + 1e-05))
                    
                task_obs = self._compute_task_obs(save_buffer = False, return_img=False)
                self._track_bodies_id = temp_tracks
                self._fut_tracks, self._fut_tracks_dropout, self._traj_sample_timestep, self._num_traj_samples, self._root_height_obs, self.obs_v = temp_fut, temp_fut_drop, temp_timestep, temp_num_steps, temp_root_height_obs, temp_obs_v
                task_obs = ((task_obs - self.running_mean.float()[self_obs_size:]) / torch.sqrt(self.running_var.float()[self_obs_size:] + 1e-05))
                full_obs = torch.cat([self_obs, task_obs], dim = -1)
                full_obs = torch.clamp(full_obs, min=-5.0, max=5.0)
                if self.distill_z_model:
                    gt_z = self.encoder.encoder(full_obs)
                    gt_z = project_to_norm(gt_z, self.embedding_norm_distill)
                    if self.z_all_distill:
                        gt_action = self.decoder.decoder(gt_z)
                    else:
                        gt_action = self.decoder.decoder(torch.cat([self_obs, gt_z], dim = -1))
                else:
                    if self.has_pnn_distill:
                        _, pnn_actions = self.pnn(full_obs)  
                        # x_all = torch.stack(pnn_actions, dim=1)
                        # weights = self.composer(full_obs)
                        # gt_action = torch.sum(weights[:, :, None] * x_all, dim=1)
                        gt_action = pnn_actions[0] ### ZL Hack 
                    else:
                        gt_action = self.encoder(full_obs)
                
                if self.save_kin_info:
                    self.kin_dict['gt_action'] = gt_action.squeeze()
                    if self.use_unrealego or self.use_visibility_branch:
                        self.kin_dict['heatmaps'] = self.ref_heatmap.clone()
                        
            ############### GT-Action Debug ################
            # if "z_acc" not in self.__dict__.keys():
            #     self.z_acc = []
            # self.z_acc.append([gt_action, actions])
            # if len(self.z_acc) > 1000:
            #     import ipdb; ipdb.set_trace()
            #     import joblib;joblib.dump(self.z_acc, "z_acc_compare_2.pkl")
            ############### GT-Z ################
            
            ################ GT-Action ################
            # actions = x_all[:, 3]  # Debugging
            # actions = gt_action ;  print("debugging!!!") # Debugging

        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        if flags.server_mode:
            dt = time.time() - t_s
            print(f'\r {1/dt:.2f} fps', end='')
            
        # dt = time.time() - t_s
        # self.fps.append(1/dt)
        # print(f'\r {np.mean(self.fps):.2f} fps', end='')
        

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)
