# Event Visual Inertial Odometry
This code implements the Event based Visual Inertial Odometry pipeline described in the paper (still in progress): EVIO by Théo Hermann and Juan Andrade-Cetto developed at the  Institut de Robòtica i Informàtica Industrial  in Barcelona-Spain.

This repository present the architecture of the proposed Deep Neural Network and a dedicated Training Framework.
A separate repository will be created to be able to use the network for real-time inference on a robotic system.

The code can be used with our own provided dataset or UZH-FPV if you dont have an event camera available. This dataset contains recording from rgb camera, imu and event camera mounted on drones. The scenes were recorded at different speeds (from regular to very fast motion) and challenging lighting conditions.

This is research code still being developed and requiring work to be a stable application.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Installation Steps](#installation-steps)
4. [Usage or Getting Started](#usage-or-getting-started)
   - [Basic Usage](#basic-usage)
   - [Advanced Features](#advanced-features)
5. [Configuration Options](#configuration-options)
6. [Contributing](#contributing)
7. [License](#license)
8. [Credits or Acknowledgments](#credits-or-acknowledgments)

## Introduction or Overview

Event Visual Inertial Odometry is an innovative solution designed to revolutionize the way robotic systems, particularly highly dynamic drones, perceive and navigate their environment. By intelligently integrating data from event cameras, RGB cameras, and IMUs (Inertial Measurement Units), EVIO achieves precise 6DoF estimation of a drone's position, even in the most challenging conditions.

Traditional odometry methods often fails during rapid maneuvers or in environments where accurate, high-frequency state estimation is critical. EVIO addresses these challenges, leveraging a multimodal neural network architecture that significantly enhances accuracy and performance. Our unique approach to data fusion involves combining RGB and event camera data through REFusion, an early fusion method that capitalizes on their complementary nature. This is followed by an integration of IMU data, utilizing Wavenet encoder features.

Designed for drones that operate in highly dynamic scenarios. Whether navigating through though environments or executing intricate flight patterns, EVIO empowers drones to achieve their full potential, opening new possibilities in autonomous systems.

## Features

- **Event Camera Integration:** Use event camera data for robust state estimation, crucial for navigating during dynamic movements where traditional sensors fall short.

- **Modular Architecture:** Designed for flexibility, allowing easy updates or changes in modalities and fusion techniques to meet diverse application needs.

- **Compatibility with UZH FPV Dataset:** Ensures seamless integration and benchmarking with an available drone flight data collection, facilitating development and testing.

- **Optimized for Dynamic Scenarios:** Tailored for high-speed and complex maneuvers, providing drones with improved accuracy and reliability in navigation.


## Installation

### Prerequisites

- pytorch
- torchvision
- mlflow

To do: add the procedure to verify the compatibility of torch and torchvision based on GPU architecture.

### Installation Steps



## Usage or Getting Started

## Download the relevant dataset

To be able to use this training framework, the user must first be sure to have downloaded the relevant datasets.
Developed to be compatible with IRI-Borinot and UZH FPV

To Do: add the instructions to preprocess the data to be compatible for training.

### Basic Usage

To do: provide the requirements.txt
source the conda environment

Set the environment variables by running

    . experiments/prepare_session.sh 0

To run the default training script:

    python src/train.py --exp_dir experiments/model1/basic_settings

### Advanced Features

Create new architectures by modifying the config files

To do: explain the process of modifying existing config to create custom architectures.

## Configuration Options

model 1/2/3
different type of modality as input
type of backbone being used for each modality
depth of the different neural nets
fusion strategy
Loss function
Decoder

## Contributing

Content here.

## License

Content here.

## Credits or Acknowledgments

Content here.



## This project is based on: IRI-DL

IRI deep learning project example. This repo contains the code structure I use for all my research. I designed it with the purpose of having a generic framework in which new research ideas could be quickly evaluated. For this particular repo, as an example, the framework has been configured to solve object classification. 

If you have any doubt do not hesitate to contact me at `apumarola@iri.upc.edu`.

#### 0. System
Upgrade system:
```
sudo apt-get update
sudo apt-get upgrade
```
#### 1. Nvidia Driver 
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo ubuntu-drivers devices
sudo apt install nvidia-driver-440
sudo reboot
```
#### 2. MiniConda
1. Download miniconda from the oficial [website](https://conda.io/miniconda.html). (Recommended: Python 3.* , 64-bits)
2. Install miniconda. (Recommended: use predefined paths and answer yes whenever you are asked yes/no)
    ```
    bash ~/Downloads/Miniconda3-latest-Linux-x86_64.sh
    ```
#### 3. Dependencies
1. Create and activate conda environment for the project
    ```
    conda create -n IRI-DL
    source activate IRI-DL
    ```
2. Install Pytorch
    ```
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    ```
3. Install other rand dependencies
    ```
    conda install matplotlib opencv pillow scikit-learn scikit-image cython tqdm
    pip install tensorboardX
    ```
4. Deactivate environment
    ```
    conda deactivate
    ```
#### 4. Tensorboard
1. Create and activate conda environment for tensorboard: 
    ```
    conda create -n tensorboard python=3.6
    source activate tensorboard
    ```
2. Install Tensorflow CPU
    ```
    pip install tensorflow
    ```
3. Deactivate environment
    ```
    conda deactivate
    ```
## Set code
Simply clone the repo:
```
cd /path/to/desired/folder/
git clone https://github.com/albertpumarola/IRI-DL.git
```

## Set IDE
1. Install PyCharm *Professional* from the official [website](https://www.jetbrains.com/pycharm/download/#section=linux) using your academic email.
2. Open Project: `Open->path/to/repo/`
3. Create desktop entry: `Tools->Create Desktop Entry...`
4. Set interpreter. In `File->Settings->Project: IRI-DL->Project Interpreter->gear->Add->Conda Environment->Existing environment:` set path to the created environment `~/miniconda3/envs/IRI-DL/bin/python`.


## Run train
1. Add configuration. In the top right `Add Configuration...->+->Python`. Introduce:
    * Name: `train`
    * Script Path: `path/to/repo/src/train.py`
    * Parameters: `--exp_dir experiments/model1/basic_settings`
    * Environment Variables:
        * `PYTHONUNBUFFERED 1`
        * `OMP_NUM_THREADS 4`
        * `CUDA_VISIBLE_DEVICES 0`
    * Python Interpreter: `Python 3.7(IRI-DL)`
    * Working Directory: `path/to/repo/`
    
    <p align="center">
      <img src="readme_resources/set_train.png" width="600" />
    </p>

2. To run train simply press play button. If you prefer running in terminal you can launch 
    ```
    . experiments/prepare_session.sh 0
    ``` 
    to set environment variables and then run 
    ```
    python src/train.py --exp_dir experiments/model1/basic_settings
    ```
3. To visualize. In a new terminal run:
    ```
    source activate tensorboard
    tensorboard --logdir path/to/repo/experiments/model1/basic_settings/
    ```
5. To run other experiments simply change the experiment dir (e.g. `experiments/model1/with_vgg_lower_lr`)

<p align="center">
  <img src="readme_resources/plot.png" width="400" />
  <img src="readme_resources/img.png" width="400" /> 
</p>

## Run test
1. Add configuration. In the top right `Add Configuration...->+->Python`. Introduce:
    * Name: `test`
    * Script Path: `path/to/repo/src/test.py`
    * Parameters: `--exp_dir experiments/model1/basic_settings`
    * Environment Variables:
        * `PYTHONUNBUFFERED 1`
        * `OMP_NUM_THREADS 1`
        * `CUDA_VISIBLE_DEVICES 0`
    * Python Interpreter: `Python 3.7(IRI-DL)`
    * Working Directory: `path/to/repo/`
2. To run test simply press play button. If you prefer running in terminal you can launch
    ```
    . experiments/prepare_session.sh 0
    ```
    to set environment variables and then run 
    ```
    python src/test.py --exp_dir experiments/model1/basic_settings
    ```
    results will be store in `experiments/model1/basic_settings/test`
3. To test other experiments simply change the experiment dir (e.g. `experiments/model1/with_vgg_lower_lr`)
