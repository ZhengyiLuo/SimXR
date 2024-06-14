[Repo still under construction]

# Real-Time Simulated Avatar from Head-Mounted Sensors

Official implementation of CVPR 2024 highlight paper: "Real-Time Simulated Avatar from Head-Mounted Sensors".


[[paper]](https://arxiv.org/abs/2403.06862) [[website]](https://zhengyiluo.github.io/SimXR/) 

<div float="center">
  <img src="assets/simxr_teaser.gif" />
</div>



## Data 
(coming soon)

## Evaluation 

Evaluate Aria models: 

```

python phc/run_hydra.py  exp_name=simxr_aria env=env_simxr_aria learning=im_simxr env.motion_file=sample_data/Apartment_release_decoration_skeleton_seq139_1WM103600M1292_0_2766_0_395.pkl  robot.box_body=False  has_eval=True real_traj=True  epoch=-1 test=True env.num_envs=1  headless=False 

```