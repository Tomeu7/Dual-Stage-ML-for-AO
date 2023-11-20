# Dual Stage ML for AO
Implementation of Dual-stage control with reinforcement and supervised learning for adaptive optics with a pyramid wavefront sensor.

## Visualization of the RL agent working with SR, action, reward and two channels of the state

<div align="center">
    <img src="https://github.com/Tomeu7/Dual-Stage-ML-for-AO/blob/main/img/visualisation.gif" width="600">
</div>

## Requirements

+ Anaconda installation with
++ Gym
++ Pytorch
++ Compass (https://github.com/ANR-COMPASS/shesha).
+ CUDA 11.6
We did the experiments COMPASS 5.2.1 and CUDA 11.6.

## Installation

For installation we provide two options:

### Option 1: install via docker (requirements: docker)

We provide an installation tutorial with docker. First build the docker image with:

```
sudo docker build -t dual-stage-ml-for-ao .
```

Now run a container with:

```
sudo docker run -it --gpus all --rm -v $HOME/test-repository-for-paper:/test-repository-for-paper dual-stage-ml-for-ao /bin/bash
```

### Option 2: install via anaconda (requirements:anaconda)

After anaconda installation you can create a new environment with python 3.8.13

```
conda env create -f environment.yml
```

Ensure CUDA version is 11.6. Otherwise, you might need to change CUDA version or COMPASS version.


## Directory structure

```
project
│   README.md
└───src
│   └───config.py # Configuration for environment and agent
│   └───rl_agent # RL folder
│       └───models.py # Helper functions and preprocessing
│       └───sac.py # Soft Actor Critic implementation
│       └───utils.py # Helper functions for the RL agent
│   └───unet # U-Net folder
│       └───dataset.py # dataset functionalities that will be used when training the U-Net
│       └───unet.py # U-Net code
│   └───mains # Code to run
│       └───mains_rl # RL folder that you can execute
│                  └───main.py # Basic agent with default functionalities
│                  └───helper.py # Helper functions for the RL closed loop
│       └───mains_unet # U-Net folder that you can execute
│                  └───main_closed_loop_unet.py # Closed loop with the U-Net
│                  └───main_recollect_data.py # Recollect data for the U-Net
│                  └───main_train.py # Train the U-Net
└───shesha # Shesha package to execute COMPASS with python
│       └───Supervisor # Scripts to communicate with COMPASS.
│       │       │RlSupervisor.py  Supervisor modification for a RL problem.
│       │       │...
│       │   ...
└───data # Folder of parameter files for different simulations and U-Net models
```

## Usage

### 1. Training Unet

First recollect data:

```
python src/mains/mains_unet/main_recollect_data.py --parameter_file "pyr_40x40_8m_M9_n3" --data_size 200000 --path_to_data "path_to_save_dir"
```

Then train:

```
python src/mains/mains_unet/main_train.py --experiment_name "test" --data_dir "path_to_save_dir" --data_name "name_of_the_folder_in_data_dir" --save_dir "path_to_save_dir" --use_voltage_as_phase
```

### 2. Testing Unet

Once trained, you can test the U-Net either in closed loop:

```
# Closed loop
python src/mains/mains_unet/main_closed_loop_unet.py --parameter_file "pyr_40x40_8m_M9_n3.py" --unet_name "40_net_Final_dataset_g9_normalization_noise_L1_relative.pth"
```

### 3. Training RL


```
# Testing the UNet+Linear in the same setting as RL
python src/mains/mains_rl/main.py --seed 1234 --parameter_file "pyr_40x40_8m_M9_n3.py" --r0 "0.12" --experiment_name "test_experiment_non_linear" --number_of_modes_filtered 100 --device 0 --control_tt --unet_name "40_net_Final_dataset_g9_normalization_noise_L1_relative.pth" --controller_type "UNet+Linear"
# Training RL
python src/mains/mains_rl/main.py --seed 1234 --parameter_file "pyr_40x40_8m_M9_n3.py" --r0 "0.12" --experiment_name "test_experiment_rl" --number_of_modes_filtered 100 --device 0 --number_of_previous_s_dm_tt=3 --number_of_previous_s_dm=3 --s_dm_tt --s_dm_residual_non_linear --s_dm_residual_non_linear_tt --mode "correction" --control_tt --unet_name "40_net_Final_dataset_g9_normalization_noise_L1_relative.pth" --controller_type "RL"

```

This work has been a collaboration of Barcelona Supercomputing Center, Paris Observatory and Universitat Politècnica de Catalunya for the RisingSTARS project.

<div align="center">
  <img src="https://github.com/Tomeu7/Dual-Stage-ML-for-AO/blob/main/img/Image1.png" width="200" />
  <img src="https://github.com/Tomeu7/Dual-Stage-ML-for-AO/blob/main/img/Image2.png" width="200" />
  <img src="https://github.com/Tomeu7/Dual-Stage-ML-for-AO/blob/main/img/Image3.jpg" width="200" />
  <img src="https://github.com/Tomeu7/Dual-Stage-ML-for-AO/blob/main/img/Image4.png" width="200" />
</div>

## Acknowledgments

We would like to thank user pranz24 for providing a working version of Soft Actor Critic in Pytorch in https://github.com/pranz24/pytorch-soft-actor-critic. Also the U-Net code shares structure with https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.