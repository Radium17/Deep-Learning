"""  Copyright 2021 MIT 6.S191 Introduction to Deep Learning. All Rights Reserved.# 
 
 Licensed under the MIT License. You may not use this file except in compliance
 with the License. Use and/or modification of this code outside of 6.S191 must
 reference:

 © MIT 6.S191: Introduction to Deep Learning
 http://introtodeeplearning.com
"""

%tensorflow_version 2.x
import tensorflow as tf

!pip install mitdeeplearning
import meetdeeplearning as mdl

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm # a progress bar

assert len(tf.config.list_physical_devices('GPU'))>0
