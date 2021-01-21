""" # Copyright 2021 MIT 6.S191 Introduction to Deep Learning. All Rights Reserved.
# 
# Licensed under the MIT License. You may not use this file except in compliance
# with the License. Use and/or modification of this code outside of 6.S191 must
# reference:
#
# © MIT 6.S191: Introduction to Deep Learning
# http://introtodeeplearning.com
#
"""
import tensorflow as tf

import IPython
import functools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Download and import the MIT 6.S191 package
!pip install mitdeeplearning
import mitdeeplearning as mdl

# Datasets: from CelebA and ImageNet
path_to_training_data = tf.keras.utils.get_file('train_face.h5', 'https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1')
loader = mdl.lab2.TrainingDatasetLoader(path_to_training_data)

number_of_training_examples = loader.get_train_size()
(images, labels) = loader.get_batch(100)

# Examining the CelebA training dataset
#@title Change the sliders to look at positive and negative training examples! { run: "auto" }

face_images = images[np.where(labels==1)[0]]
not_face_images = images[np.where(labels==0)[0]]

idx_face = 23 #@param {type:"slider", min:0, max:50, step:1}
idx_not_face = 6 #@param {type:"slider", min:0, max:50, step:1}

plt.figure(figsize=(5,5))
plt.subplot(1, 2, 1)
plt.imshow(face_images[idx_face])
plt.title("Face"); plt.grid(False)

plt.subplot(1, 2, 2)
plt.imshow(not_face_images[idx_not_face])
plt.title("Not Face"); plt.grid(False)

# Define the CNN model
n_filters = 12 #base number of convolutional filters

def make_standard_classifier(n_outputs=1):
  Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu')
  BatchNormalization = tf.keras.layers.BatchNormalization
  Flatten = tf.keras.layers.Flatten
  Dense = functools.partial(tf.keras.layers.Dense, activation='relu')
  
  model = tf.keras.Sequential([
    Conv2D(filters=1*n_filters, kernel_size=5, strides=2),
    BatchNormalization(),
    
    Conv2D(filters=2*n_filters, kernel_size=5, strides=2),
    BatchNormalization(),
    
    Conv2D(filters=4*n_filters, kernel_size=3, strides=2),
    BatchNormalization(),
    
    Conv2D(filters=6*n_filters, kernel_size=3, strides=2),
    BatchNormalization(),
    
    Flatten(),
    Dense(512),
    Dense(n_outputs, activation=None), 
    
  ])
  return model

standard_classifer = make_standard_classifier()

# Train the standard CNN

batch_size = 32
num_epochs = 2
learning_rate = 5e-4

optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_history = mdl.util.LossHistory(smoothing_factor=0.99)
plotter = mdl.util.PeriodicPlotter(sec=2, scale='semilogy')
if hasattr(tqdm, '_instances'): tqdm._instances.clear()

@tf.function  
def standard_train_step(x, y):
  with tf.GradientTape() as tape:
    logits = standard_classifier(x)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(y, logits)
    
  # Backpropagation
  grads = tape.gradient(loss, standard_classifier.trainable_variables)
  optimizeer.apply_gradients(zip(grads, standard_classifier.trainable_variables))
  return loss

# The training loop
for epoch in range(num_epochs):
  for idx in tqdm(range(loader.get_train_size()//batch_size)):
    x, y = loader.get_batch(batch_size)
    loss = standard_tarin_step(x, y)
    
    loss_history.append(loss.numpy().mean())
    plotter.plot(loss_history.get())

# Evaluation 
(batch_x, batch_y) = loader.get_batch(5000)
y_pred_standard = tf.round(tf.nn.sigmoid(standard_classifier.predict(batch_x)))
acc_standard = tf.reduce_mean(tf.cast(tf.equal(betch_y, y_pred_standard), tf.float32))

print("Standard CNN accuracy on (potentially biased training set: {:.4f})".format(acc_standard.numpy()))

# Load test dataset and plot examples
test_faces = mdl.lab2.get_test_faces()
keys = ["Light Female", "Light Male", "Dark Female", "Dark Male"]
for group, key in zip(test_faces, keys):
  plt.figure(figsize=(5,5))
  plt.imshow(np.hstack(group))
  plt.title(key, fontsize=15)
  
std_classifier_logits = [standard_classifier(np.array(x, dtype=np.float32)) for x in test_faces]
std_classifier_probs = tf.squeeze(tf.sigmoid(std_classifier_logits))

# Plot the prediction accuracies per demodraphic
xx = range(len(keys))
yy = std_classifier_probs.numpy().mean(1)
plt.bar(xx, yy)
plt.xticks(xx, keys)
plt.ylim(max(0,yy.min()-yy.ptp()/2.), yy.max()+yy.ptp()/2.)
plt.title("Standard classifier predictions")

"""  In learning the latent space, we constrain the means and standard deviations to approximately follow a unit Gaussian. 
     Recall that these are learned parameters, and therefore must factor into the loss computation, and that the decoder portion of 
     the VAE is using these parameters to output a reconstruction that should closely match the input image, which also must factor 
     into the loss. What this means is that we'll have two terms in our VAE loss function:

1.  **Latent loss ($L_{KL}$)**: measures how closely the learned latent variables match a unit Gaussian and is defined by the Kullback-Leibler (KL) divergence.
2.  **Reconstruction loss ($L_{x}{(x,\hat{x})}$)**: measures how accurately the reconstructed outputs match the input and is given by the $L^1$ norm of
    the input image and its reconstructed output.
"""




