mkdir sample_data
mkdir -p output output/HumanoidIm/ output/HumanoidIm/simxr_aria
gdown https://drive.google.com/uc?id=1bLp4SNIZROMB7Sxgt0Mh4-4BLOPGV9_U -O  sample_data/ # filtered shapes from AMASS
gdown https://drive.google.com/uc?id=1arpCsue3Knqttj75Nt9Mwo32TKC4TYDx -O  sample_data/ # all shapes from AMASS
gdown https://drive.google.com/uc?id=1fFauJE0W0nJfihUvjViq9OzmFfHo_rq0 -O  sample_data/ # sample standing neutral data.
gdown https://drive.google.com/uc?id=1uzFkT2s_zVdnAohPWHOLFcyRDq372Fmc -O  sample_data/ # amass_occlusion_v3
gdown https://drive.google.com/uc?id=1oQEDHzwZ3s20WbHyWfOyjcU2ho81ewHA -O  sample_data/ # aria sample data 
gdown https://drive.google.com/uc?id=107i5YyM_2a2MkPTfJ04z-PGpREa02fVM -O  output/HumanoidIm/simxr_aria/Humanoid.pth # aria model 

