"""# Copyright 2021 MIT 6.S191 Introduction to Deep Learning. All Rights Reserved.
# 
# Licensed under the MIT License. You may not use this file except in compliance
# with the License. Use and/or modification of this code outside of 6.S191 must
# reference:
#
# © MIT 6.S191: Introduction to Deep Learning
# http://introtodeeplearning.com
"""

# Install some dependencies for visualizing the agents
!apt-get install -y xvfb python-opengl x11-utils > /dev/null 2>&1
!pip install gym pyvirtualdisplay scikit-video 2>&1

%tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import base64, io, time, gym
import IPython, functools
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

!pip install mitdeeplearning
import mitdeeplearning as mdl

### PART 1: Cartpole

# Define the Cartpole environment and agent
# Define the agent's memory
# Reward function
# Learning algorithm
# Cartpole training

### PART 2: Pong

