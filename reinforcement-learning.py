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
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)
cartpole_model = create_model()

# to track the progress
smoothed_reward = mdl.util.LossHistory(smoothing_factor = 0.9)
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Rewards')
if hasattr(tqdm, '_instances'): tqdm._instances.clear()
for i_episode in range(500):
  plotter.plot(smoothed_reward.get())
  
  observation = env.reset()
  memory.clear()
  
  while True:
    action = choose_action(cartpole_model, observation)
    next_observation, reward, done, info = env.step(action)
    memory.add_to_memory(observation, action, reward)
    
    # is the episode over? did you crash or did you so well that you're done?
    if done:
      total_reward = sum(memory.rewards)
      smoothed_reward.append(total_reward)
      
      # initiate training - we don't know anything about how the agent
      # is doing until it has crashed 
      train_step(cartpole_model, optimizer,
                observations = np.vstack(memory.observations),
                actions = np.array(memory.actions),
                discounted_rewards = discount_rewards(memory.rewards))
      
      memory.clear()
      break
    observation = next_observation
    
    # to watch how the agent did
    saved_cartpole = mdl.lab3.save_video_of_model(cartpole_model, "CartPole-v0")
    mdl.lab3.play_video(saved_cartpole)

### PART 2: Pong

def create_pong_env():
  return gym.make("Pong-v0", frameskip=5)
env = create_pong_env()
env.seed(1) 

# In the case of Pong our observations are the individual video frames(i.e. images)
# Thus, the observations are 210x160 RGB
print("Environment has observation space =", env.observation_space)

# 6 actions: no-op, move right/left, fire, fire right/left
n_actions = env.action_space.n
print("Number of possible actions that the agent can choose from =", n_actions)

# Define the Pong agent
Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense

def create_pong_model():
  model = tf.keras.models.Sequential([
    Conv2D(filters=32, kernel_size=5, stride=2),
    Conv2D(filters=48, kernel_size=5, stride=2),
    Conv2D(filters=64, kernel_size=3, stride=2),
    Conv2D(filters=64, kernel_size=3, stride=2),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=n_actions, activation='None')
  ])
  return model

pong_model = create_pong_model()

# Pong reward function
def discount_rewards(rewards, gamma=0.99):
  discounted_rewards = np.zeros_like(rewards)
  R = 0
  for t in reversed(range(0, len(rewards))):
    # reset the sum if the reward is not 0 (the game has ended)
    if rewards[t] != 0:
      R = 0
    R = R*gamma + rewards[t]
    discounted_rewards[t] = R
    
  return normalize(discounted_rewards)

# To visualize a single observation before and after pre-processing
observation = env.reset()
for i in range(30):
  action = np.random.choice(n_actions)
  observation, _, _, _ = env.step(action)
observation_pp = mdl.lab3.preprocess_pong(observation)

f = plt.figure(figsize=(10,3))
ax = f.add_subplot(121)
ax2 = f.add_subplot(122)
ax.imshow(observation); ax.grid(False)
ax2.imshow(np.squeeze(observation_pp)); ax2.grid(False);
plt.title('Preprocessed Observation')

# To capture the dynamics, i.e. how the environment changes
next_observation, _, _, _ = env.step(np.random.choice(n_actions))
diff = mdl.lab3.pong_change(observation, next_observation)

f, ax = plt.subplot(1, 3, figsize=(15,15))
for a in ax:
  a.grid(False)
  a.axis("off")
ax[0].imshow(observation); ax[0].set_title('Previous Frame');
ax[1].imshow(next_observation); ax[1].set_title('Current Frame');
ax[2].imshow(np.squeeze(diff)); ax[2].set_title('Difference (Model Input)');

def collect_rollout(batch_size, env, model, choose_action):
  memories = []
  for b in range(batch_size):
    memory = Memory()
    next_observation = env.reset()
    previous_frame = next_observation
    done = False # tracks whether the episode (game) is done or not
    
    while not done:
      current_frame = next_observation
      frame_diff = mdl.lab3.pong_change(previous_frame, current_frame)
      action = choose_action(model, frame_diff)
      next_observation, reward, done, info = env.step(action)
      memory.add_to_memory(frame_diff, action, reward)
      previous_frame = current_frame
    
    memories.append(memory)
  return memories

# Rollout with untrained Pong model
test_model = create_pong_model()
single_batch_size = 1
memories = collect_rollout(single_batch_size, env, test_model, choose_action)
rollout_video = mdl.lab3.save_video_of_memory(memories[0], "Pong-Random-Agent.mp4")
mdl.lab3.play_video(rollout_video)

# Training Pong
learning_rate = 1e-3
MAX_ITERS = 1000
batch_size = 5

pong_model = create_pong_model()
optimizer = tf.keras.optimizers.Adam(learning_rate)
iteration = 0

smoothed_reward = mdl.utils.LossHistory(smoothing_factor=0.9)
smoothed_reward.append(0) # start the reward at zero for baseline comparison
plotter = mdl.util.PeriodicPlotter(sec=15, xlabel='Iterations', ylabel='Win Percentage (%)')

# To parallelize batches, we need to make multiple copies of the environment
envs = [create_pong_env() for _ in range(batch_size)]

games_to_win_episode = 21 # this is set by OpenAI gum and cannot be changed
while iteration < MAX_ITERS:
  
  plotter.plot(smoothed_reward.get())
  tic = time.time()
  
  # By default, RL agent uses serial batch processing
  # memories = collect_rollout(batch_size, env, pong_model, choose_action)
  
  # Parallelized version. 
  memories = mdl.lab3.parallelized_collect_rollout(batch_size, envs, pong_model, choose_action)
  print(time.time() - tic)
  
  batch_memory = aggregate_memories(memories)
  total_wins = sum(np.array(batch_memory.rewards)==1)
  total_games = sum(np.abs(np.array(batch_memory.rewards)))
  win_rate = total_wins / total_games
  smoothed_reward.append(100 * win_rate)
  
  # Training
  train_step(
      pong_model,
      optimizer,
      observations = np.stack(batch_memory.observations, 0),
      actions = np.array(batch_memory.actions),
      discounted_rewards = discount_rewards(batch_memory.rewards)
  )
  
  if iteration % 100 == 0:
    mdl.lab3.save_video_of_model(pong_model, "Pong-v0",
                                suffix="_"+str(iteration))
  iteration += 1
  
  latest_pong = mdl.lab3.save_video_of_model(
      pong_model, "Pong-v0", suffix="_latest")
  mdl.lab3.play_vdeo(latest_pong, width=400)
