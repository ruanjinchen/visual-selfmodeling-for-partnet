# Visual Self-Modeling for articulate objects in PartNet Mobility Dataset

## Overview
This repo contains the PyTorch implementation for paper "Full-Body Visual Self-Modeling of Robot Morphologies".


## Content

- [Visual Self-Modeling for articulate objects in PartNet Mobility Dataset](#visual-self-modeling-for-articulate-objects-in-partnet-mobility-dataset)
  - [Overview](#overview)
  - [Content](#content)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [About Configs and Logs](#about-configs-and-logs)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [License](#license)
  - [Reference](#reference)


## Installation

This code has been tested on Windows11 with CUDA 13.0 and Ubuntu 22.04 with CUDA 12.4.  
Create a python3.12 virtual environment and install the dependencies. I recommend using PyTorch's official installation instructions.

```
conda create -y -n vsm python=3.12

# PyTorch + cu124 Official Index
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision

# PyTorch + cu130 Official Index
pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch torchvision

pip install -r requirements.txt
```

## Data Preparation

Run the following commands to generate the simulated data in Pybullet. Needs PartNet Datasets!

```
python make_partnet_dataset_for_vsm.py \
  --urdf-dir assets/102074 \
  --out-dir data/pliers_2074 \
  --num 500 \
  --scale-factor 1.6780842542648315 \
  --mesh-format obj \
  --pcd-format ply \
  --pcd-points 400000 \
  --depth-width 640 --depth-height 480 --depth-fov 75 \
  --cam-dist-mul 3.2 \
  --near-mul 0.05 --far-mul 6.0 \
  --extra-views 24 \
  --seed 1

python make_partnet_dataset_for_vsm.py \
  --urdf-dir assets/10893 \
  --out-dir data/scissors_10893 \
  --num 500 \
  --scale-factor 0.9951514005661011 \
  --mesh-format obj \
  --pcd-format ply \
  --pcd-points 4000000 \
  --depth-width 1280 --depth-height 720 --depth-fov 75 \
  --cam-dist-mul 3.2 \
  --near-mul 0.05 --far-mul 6.0 \
  --extra-views 24 \
  --seed 1

python make_partnet_dataset_for_vsm.py \
  --urdf-dir assets/101863 \
  --out-dir data/eyeglasses_101863 \
  --num 500 \
  --scale-factor 1.0354602336883545 \
  --mesh-format obj \
  --pcd-format ply \
  --pcd-points 4000000 \
  --depth-width 1280 --depth-height 720 --depth-fov 75 \
  --cam-dist-mul 3.2 \
  --near-mul 0.05 --far-mul 6.0 \
  --extra-views 24 \
  --seed 1
```
This will generate all files you needed for training.

## About Configs and Logs

Before training and evaluation, we first introduce the configuration and logging structure.

**Configs:** all the specific parameters used for training and evaluation are indicated in `./configs/state_condition/config1.yaml`. If you would like to play with other parameters, feel free to copy the existing config file and modify it. You will then just need to change the config file path in the following training steps to point to the new configuration file.

To train the self-model which also predicts the end effector position together with our visual self-model, please use `./configs/state_condition_kinematic/config1.yaml`.

To train the self-model which only predicts the end effector from scratch, without out visual self-model, please use `./configs/state_condition_kinematic_scratch/config1.yaml`.

If you save the data to other directories, please make sure the `data_filepath` argument in each config file points to the correct path.

**Logs:** both the training and evaluation results will be saved in the log folder for each experiment. The log folders will be located under `./scripts` folder. The last digit in the logs folder indicates the random seed. Inside the logs folder, the structure and contents are:

    ```
    \logs_True_False_False_image_conv2d-encoder-decoder_True_{output_representation}_{seed}
        \lightning_logs
            \checkpoints          [saved checkpoint]
            \version_0            [training stats]
        \predictions              [complete predicted meshes before normalization]
        \predictions_denormalized [complete predicted meshes after normalization]
    ```

## Training

To train our visual self-model, run the following command.

```
cd scripts;
CUDA_VISIBLE_DEVICES=0 python ../main.py ../configs/state_condition/config1.yaml NA;
```

To use our pre-trained self-model to train a small network to predict end-effector position, run the following command. For this step, please uncomment the validation code in `models.py` (line 143-158, line 202-204, and line 225-231). Please only uncomment then for this particular step.

```
cd scripts;
CUDA_VISIBLE_DEVICES=0 python ../main.py ../configs/state_condition_kinematic/config1.yaml kinematic ./logs_state-condition_new-global-siren-sdf_1/lightning_logs/version_0/checkpoints/;
```

To train the baseline model that predicts end-effector position from scratch, without using our visual self-model, run the following command. For this step, please uncomment the validation code in `models.py` (line 143-158, line 202-204, and line 225-231). Please only uncomment then for this particular step.

```
CUDA_VISIBLE_DEVICES=0 python ../main.py ../configs/state_condition_kinematic_scratch/config1.yaml kinematic-scratch NA;
```

## Evaluation

To evaluate the predicted meshes and compare with baselines, run the following commands.

```
cd scripts;
CUDA_VISIBLE_DEVICES=0 python ../eval.py ../configs/state_condition/config1.yaml ./logs_state-condition_new-global-siren-sdf_1/lightning_logs/version_0/checkpoints/ eval-state-condition;

cd utils;
python eval_mesh.py ../configs/state_condition/config1.yaml model;
python eval_mesh.py ../configs/state_condition/config1.yaml nearest-neighbor;
python eval_mesh.py ../configs/state_condition/config1.yaml random;

CUDA_VISIBLE_DEVICES=0 python ../eval.py ../configs/state_condition_kinematic/config1.yaml ./logs_state-condition-kinematic_new-global-siren-sdf_1/lightning_logs/version_0/checkpoints/ eval-kinematic ./logs_state-condition_new-global-siren-sdf_1/lightning_logs/version_0/checkpoints/;

CUDA_VISIBLE_DEVICES=4 python ../eval.py ../configs/state_condition_kinematic_scratch/config1.yaml ./logs_state-condition-kinematic-scratch_new-global-siren-sdf_1/lightning_logs/version_0/checkpoints/ eval-kinematic;
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Reference

- https://github.com/vsitzmann/siren
- https://github.com/autonomousvision/occupancy_networks/