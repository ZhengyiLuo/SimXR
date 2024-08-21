
# Real-Time Simulated Avatar from Head-Mounted Sensors

Official implementation of CVPR 2024 highlight paper: "Real-Time Simulated Avatar from Head-Mounted Sensors".


[[paper]](https://arxiv.org/abs/2403.06862) [[website]](https://zhengyiluo.github.io/SimXR/) 

<div float="center">
  <img src="assets/simxr_teaser.gif" />
</div>

## News 🚩

[August 20, 2024] Data released!

[August 5, 2024] Evaluation code released!

[May 11, 2024] Skeleton code Released!


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
### Aria 

Processed Aria sequences can be found here for training and evaluation: [[Train]](https://drive.google.com/drive/folders/1ZsT4sgz3NUmpoMqcR35KJ-hFfuknrVQi?usp=drive_link) [[Test]](https://drive.google.com/drive/folders/10L8tARGzShPwzG1aJM3fPzIxuarEBAKW?usp=drive_link)

### Quest 2
Processed real-world sequences can be found here for evaluations: [[Test]](https://drive.google.com/drive/folders/1z6cviNR624UERdi8YrAMCyHbjMitsZO9?usp=sharing)

Processed synthetic sequences can be found here for training: [[Train]](https://drive.google.com/drive/folders/1jMld_d6JmyNkq0w1mBWH5nSts9tsO47b?usp=drive_link) [[Test]](https://drive.google.com/drive/folders/1RACtEleG5saxvjyt1KRe-p0jA4WmAGMS?usp=drive_link)

### Splitting Data 
After downloading the data, you can split the data into training and testing data using the following command: 

``` 
python scripts/data_process/split_data_syn.py 
python scripts/data_process/split_data_aria.py 

```


## Evaluation 

Evaluate Aria models: 

```

python phc/run_hydra.py  exp_name=simxr_aria env=env_simxr_aria learning=im_simxr env.motion_file=sample_data/Apartment_release_decoration_skeleton_seq139_1WM103600M1292_0_2766_0_395.pkl  robot.box_body=False  env.cycle_motion=False has_eval=True real_traj=True  epoch=-1 test=True env.num_envs=1  headless=False no_virtual_display=True

```


Evaluate Quest 2 models using real-world sequences: 
```
python phc/run_hydra.py  exp_name=simxr_quest env=env_simxr_quest learning=im_simxr env.motion_file=sample_data/capture00.pkl  robot=quest_humanoid  env.cycle_motion=False has_eval=True real_traj=True  epoch=-1 test=True env.num_envs=1  headless=False  no_virtual_display=True
```

Evaluate Quest 2 motion imitator
```
python phc/run_hydra.py  exp_name=phc_prim_quest env=env_im_quest learning=im_quest env.motion_file=sample_data/capture00.pkl  robot=quest_humanoid  env.cycle_motion=False has_eval=True  epoch=-1 test=True env.num_envs=1  headless=False  no_virtual_display=True

```



## Training

Train Aria models: 

```
python phc/run_hydra.py  exp_name=simxr_aria env=env_simxr_aria learning=im_simxr env.motion_file=[Inerst Motion Data]  robot.box_body=False
```

Train Quest 2 models: 

```
python phc/run_hydra.py  exp_name=simxr_quest env=env_simxr_quest learning=im_simxr env.motion_file=[insert synthetic data location]  robot=quest_humanoid
```


## Citation
If you find this work useful for your research, please cite our paper:
```
@InProceedings{Luo_2024_CVPR,
    author    = {Luo, Zhengyi and Cao, Jinkun and Khirodkar, Rawal and Winkler, Alexander and Kitani, Kris and Xu, Weipeng},
    title     = {Real-Time Simulated Avatar from Head-Mounted Sensors},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {571-581}
}   
```

Also consider citing these prior works that are used in this project:

```
@inproceedings{Luo2023PerpetualHC,
    author={Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
    title={Perpetual Humanoid Control for Real-time Simulated Avatars},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2023}
}         

@inproceedings{rempeluo2023tracepace,
    author={Rempe, Davis and Luo, Zhengyi and Peng, Xue Bin and Yuan, Ye and Kitani, Kris and Kreis, Karsten and Fidler, Sanja and Litany, Or},
    title={Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}     

@inproceedings{Luo2022EmbodiedSH,
  title={Embodied Scene-aware Human Pose Estimation},
  author={Zhengyi Luo and Shun Iwase and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{Luo2021DynamicsRegulatedKP,
  title={Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation},
  author={Zhengyi Luo and Ryo Hachiuma and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}

```


## References
This repository is built on top of the following amazing repositories:
* Main code framework is from: [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
* Main code framework is from: [PHC](https://github.com/ZhengyiLuo/PHC)
* SMPL models and layer is from: [SMPL-X model](https://github.com/vchoutas/smplx)

Please follow the lisence of the above repositories for usage. 
