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

# Instantiate the Cartpole environment 
env = gym.make("CartPole-v0")
env.seed(1)

# In this Cartpole environment our observations are Cart position & velocity; 
# Pole angle & rotation rate
n_observations = env.observation_space
print("Environment has observation space = ", n_observations)

# the agent can move either right ot left
n_actions = env.action_space.n
print("Number of possible actions that the agent can choose from=", n_actions)

# Define the Cartpole agent
def create_cartpole_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units = 32, activation = 'relu'),
    tf.keras.layers.Dense(units = n_actions, activation = None)
  ])
  return model

cartpole_model = create_cartpole_model()

# Define the agent's action function
def choose_action(model, observation, single = True):
  """
  model: the network that defines our agent
  observation: observation(s) which is/are fed as input to the model
  single: handle batch of observations or not
  """
  observation = np.expand_dims(observation, axis = 0) if single else observation
  logits = model.predict(observation)
  action = tf.random.categorical(logits, num_samples=1)
  action = action.numpy().flatten()
  
  return action[0] if single else action

# Define the agent's memory

class Memory:
  def __init__(self):
    self.clear()
    
  # Resets/restarts the memory buffer
  def clear(self):
    self.observations = []
    self.actions = []
    self.rewards = []
    
  def add_to_memory(self, new_observation, new_action, new_reward):
    self.observations.append(new_observation)
    self.actions.append(new_action)
    self.rewards.append(new_reward)
    
  def aggregat_memories(memories):
    batch_memory = Memory()
    for memory in memories:
      for step in zip(memory.observations, memory.actions, memory.rewards):
        batch_memory.add_to_memory(*step)
        
    return batch_memory
  
memory = Memory()
  
# Reward function
def normalize(x):
  x -= np.mean(x)
  x /= np.std(x)
  
  return x.astype(np.float32)

def discount_rewards(rewards, gamma=0.95):
  discounted_rewards = np.zeros_like(rewards)
  R = 0
  for t in reversed(range(0, len(rewards))):
    R = R*gamma + rewards[t]
    discounted_rewards[t] = R
    
  return normalize(discounted_rewards)
  
# Learning algorithm
def compute_loss(logits, actions, rewards):
  neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits = logits, labels = actions)
  
  #scale the negative log probability by the rewards
  loss = tf.reduce_mean(rewards*new_logprobs)
  
  return loss

def train_step(model, optimizer, observations, actions, discounted_rewards):
  with tf.GradientTape() as tape:
    # Forward propagate through rhe agent network
    logits = model(observations)
    
    loss = compute_loss(logits, actions, discounted_rewards)
    
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
























# Cartpole training

### PART 2: Pong

