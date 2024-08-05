[Repo still under construction]

# Real-Time Simulated Avatar from Head-Mounted Sensors

Official implementation of CVPR 2024 highlight paper: "Real-Time Simulated Avatar from Head-Mounted Sensors".


[[paper]](https://arxiv.org/abs/2403.06862) [[website]](https://zhengyiluo.github.io/SimXR/) 

<div float="center">
  <img src="assets/simxr_teaser.gif" />
</div>



## Data 
(in progress)

Processed real-world sequences can be found here for evaluations: [Google Drive](https://drive.google.com/drive/folders/1z6cviNR624UERdi8YrAMCyHbjMitsZO9?usp=sharing)

## Evaluation 

Evaluate Aria models: 

```

python phc/run_hydra.py  exp_name=simxr_aria env=env_simxr_aria learning=im_simxr env.motion_file=sample_data/Apartment_release_decoration_skeleton_seq139_1WM103600M1292_0_2766_0_395.pkl  robot.box_body=False  env.cycle_motion=False has_eval=True real_traj=True  epoch=-1 test=True env.num_envs=1  headless=False 

```


Evaluate Quest 2 models using real-world sequences: 
```

python phc/run_hydra.py  exp_name=simxr_quest env=env_simxr_quest2 learning=im_simxr env.motion_file=sample_data/capture00.pkl  robot=quest_humanoid  env.cycle_motion=False has_eval=True real_traj=True  epoch=-1 test=True env.num_envs=1  headless=False 

```


## Training

Train Aria models: 

```
python phc/run_hydra.py  exp_name=simxr_aria env=env_simxr_aria learning=im_simxr env.motion_file=[Inerst Motion Data]  robot.box_body=False
```