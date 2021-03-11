# Bomberman Reinforcement Learning

Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game 
Bomberman.

## Setup

First run:

        conda create -n bomberman python=3.8
        conda activate bomberman
        conda install scipy numpy matplotlib
        pip install scikit-learn pygame tqdm

Then follow [this](https://deeplizard.com/learn/video/IubEtS2JAiY) tutorial or just execute the following steps:
- Install tensorflow according to the [documentation](https://www.tensorflow.org/install)
- Install the [CUDA Toolkit v11.2.1](https://developer.nvidia.com/cuda-toolkit-archive)
- Install [NVIDIA cuDNN v8.1.0](https://developer.nvidia.com/cudnn) according to the [documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

Also, please create the following directories:

        cd agent_code/big_bertha_v1
        mkdir models
        mkdir tensorboard_logs

## Helper Utility

Install following helper tools for development purposes:

        pip install isort
        pip install tensorboard

Open a tensorboard after training with:

        cd agent_code/big_bertha_v1
        tensorboard --logdir tensorboard_logs
