[Repo still under construction]

# Real-Time Simulated Avatar from Head-Mounted Sensors

Official implementation of CVPR 2024 highlight paper: "Real-Time Simulated Avatar from Head-Mounted Sensors".


[[paper]](https://arxiv.org/abs/2403.06862) [[website]](https://zhengyiluo.github.io/SimXR/) 

<div float="center">
  <img src="assets/simxr_teaser.gif" />
</div>

### Dependencies
1. Create new conda environment and install pytroch:


```
conda create -n isaac python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirement.txt
```

2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym). 


3. [Optional if only inference] Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. Please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. Rename The file structure should look like this:

```

|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl

```

## Data 

Processed real-world sequences can be found here for evaluations: [Google Drive](https://drive.google.com/drive/folders/1z6cviNR624UERdi8YrAMCyHbjMitsZO9?usp=sharing)

Processed synthetic sequences can be found here for training: [Google Drive](https://drive.google.com/drive/folders/1un7F4xPy4sxPvBT_T4uYUvbWVF8Ia9-r?usp=drive_link)

## Evaluation 

Evaluate Aria models: 

```

python phc/run_hydra.py  exp_name=simxr_aria env=env_simxr_aria learning=im_simxr env.motion_file=sample_data/Apartment_release_decoration_skeleton_seq139_1WM103600M1292_0_2766_0_395.pkl  robot.box_body=False  env.cycle_motion=False has_eval=True real_traj=True  epoch=-1 test=True env.num_envs=1  headless=False no_virtual_display=True

```


Evaluate Quest 2 models using real-world sequences: 
```

python phc/run_hydra.py  exp_name=simxr_quest env=env_simxr_quest learning=im_simxr env.motion_file=sample_data/capture00.pkl  robot=quest_humanoid  env.cycle_motion=False has_eval=True real_traj=True  epoch=-1 test=True env.num_envs=1  headless=False  no_virtual_display=True

```


## Training

Train Aria models: 

```
python phc/run_hydra.py  exp_name=simxr_aria env=env_simxr_aria learning=im_simxr env.motion_file=[Inerst Motion Data]  robot.box_body=False
```
