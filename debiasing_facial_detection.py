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

""" 
In learning the latent space, we constrain the means and standard deviations to approximately follow a unit Gaussian. 
Recall that these are learned parameters, and therefore must factor into the loss computation, and that the decoder portion of 
the VAE is using these parameters to output a reconstruction that should closely match the input image, which also must factor 
into the loss. What this means is that we'll have two terms in our VAE loss function:

1.  **Latent loss ($L_{KL}$)**: measures how closely the learned latent variables match a unit Gaussian and is defined by the Kullback-Leibler (KL) divergence.
2.  **Reconstruction loss ($L_{x}{(x,\hat{x})}$)**: measures how accurately the reconstructed outputs match the input and is given by the $L^1$ norm of
    the input image and its reconstructed output.
"""

# Defining the VAE loss function
def vae_loss_function(x, x_recon, mu, logsigma, kl_weight=0.0005):
  latent_loss = 0.5*tf.reduce_sum(tf.exp(logsigma)+tf.square(mu)-1-logsigma, axis=1)  #Kullback-Leibler (KL) divergence
  reconstruction_loss = tf.reduce_sum(tf.abs(x-x_recon), axis=(1,2,3))
  vae_loss = kl_weight*latent_loss + reconstruction_loss
  
  return vae_loss

# VAE Reparametrization by sampling from an isotropic unit Gaussian
def sampling(z_mean, z_logsigma):
  """
  z_mean, z_logsigma (tensor): mean and log of std of latent distribution (Q(z|X))
  returns z(tensor): sampled latent vector
  """
  
  batch, latent_dim = z_mean.shape
  epsilon = tf.random.normal(shape=(batch, latent_dim))
  z = z_mean + tf.exp(0.5*z_logsigma)*epsilon
  
  return z

# Loss function for DB-VAE
def debiasing_loss_function(x, x_recon, y, y_logit, mu, logsigma):
  
  vae_loss = vae_loss_function(x, x_recon, mu, logsigma, kl_weight=0.0005)
  classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(y, y_logit)
  face_indicator = tf.cast(tf.equal(y,1), tf.float32)
  total_loss = tf.reduce_mean(classification_loss + face_indicator*vae_loss)
  
  return total_loss, classification_loss

# Decoder part of the dB-DAE
n_filters = 12
latent_dim = 100

def make_face_decoder_network():
  
  Conv2DTranspose = functools.partial(tf.keras.layers.Conv2DTranspose, padding='same', activation='relu')
  BatchNormalization = tf.keras.layers.BatchNormalization
  Flatten = tf.keras.layers.Flatten
  Dense = functools.partial(tf.keras.layers.Dense, activation='relu')
  Reshape = tf.keras.layers.Reshape
  
  # Build the decoder network
  decoder = tf.keras.Sequential([
    # Transform to pre-convolutional generation
    Dense(units=4*4*6*n_filters), # 4x4 feature matrix with 6N occurances
    Reshape(target_shape=(4,4,6*n_filters)),
    
    # Upscaling convolutions (inverse to encoder)
    Conv2DTranspose(filters=4*n_filters, kernel_size=3, strides=2),
    Conv2DTranspose(filters=2*n_filters, kernel_size=3, strides=2),
    Conv2DTranspose(filters=1*n_filters, kernel_size=5, strides=2),
    Conv2DTranspose(filters=3, kernel_size=5, strides=2),
  ])
  
  return decoder

# Defining and creating the DB-VAE
class DB_VAE(tf.keras.Model):
  
  def __init__(self, latent_dim):
    super(DB_VAE, self).__init__()
    self.latent_dim = latent_dim
    
    # `latent_dim` latent variables & a supervised output for the classification
    num_encoder_dims = 2*self.latent_dims + 1
    
    self.encoder = make_standard_classifier(num_encoder_dims)
    self.decoder = make_face_decoder_network()
    
  def encode(self, x):
    encoder_output = self.encoder(x)
    # classification prediction
    y_logit = tf.expand_dims(encoder_output[:,0], -1)
    # latent variable distribution parameters
    z_mean = encoder_output[:, 1:self.latent_dim+1]
    z_logsigma = encoder_output[:, self.latent_dim+1:]
    
    return y_logit, z_mean, z_logsigma
  
  # VAE reparametrization: given a mean and logsigma, sample latent variables
  def reparameterize(self, z_mean, z_logsigma):
    return sampling(z_mean, z_logsigma)
  
  # Decode the latent space and output reconstruction
  def decode(self, z):
    return self.decoder(z)
    
  def call(self, x):
    y_logit, z_mean, z_logsigma = self.encode(x)
    z = self.reparameterize(z_mean, z_logsigma)
    recon = self.decode(z)
    
    return y_logit, z_mean, z_logsigma, recon
  
  def predict(self, x):
    y_logit, _, _ = self.encode(x)
    return y_logit
  
dbvae = DB_VAE(latent_dim)

# Function to return the means for an input image batch
def get_latent_mu(images, dbvae, batch_size=1024):
  N = images.shape[0]
  mu = np.zeros((N, latent_dim))
  for start_ind in range(0, N, batch_zise):

    end_ind = min(start_ind+batch_size, N+1)
    batch = (images[start_ind:end_ind]).astype(np.float32)/255.
    _, batch_mu, _ = dbvae.encode(batch)
    mu[start_ind:end_ind] = batch_mu
    
    return mu
  
# Resampling algorithm for DB_VAE
""" Function that recomputes the sampling probabilities for images within a batch
    based on how they distribute across the training data. """

def get_training_sample_probabilities(images, dbvae, bins=10, smoothing_fac=0.001):
  print("Recomputing the sampling probabilities")
  mu = get_latent_mu(images, dbvae)
  training_sample_p = np.zeros(mu.shape[0])
  
  #consider the distribution for each latent variable
  for i in range(latent_dim):
    
    latent_distribution = mu[:,i]
    hist_density, bin_edges = np.histogram(latent_distribution, density=True, bins=bins)
    
    # find which latent bin every data sample falls in
    bin_edges[0] = -float('inf')
    bin_edges[-1] = float('inf')
    
    bin_idx = np.digitize(latent_distribution, bin_edges)
    hist_smoothed_density = hist_density + smoothing_fac
    hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)
    
    p = 1.0/(hist_smoothed_density[bin_idx-1])
    p = p / np.sum(p)
    
    training_sample_p = np.maximum(p, training_sample_p)
  
  training_sample_p /= np.sum(training_sample_p)
  
  return training_sample_p

    
# Training the DB-VAE
batch_size = 32
learning_rate = 5e-4
latent_dim = 100
num_epochs = 6

# Instantiate a new DB-VAE model and optimizer
dbvae = DB_VAE(100)
optimizer = tf.keras.optimizers.Adam(learning_rate)


# To define the training operation, we will use tf.function which is a powerful tool 
#   that lets us turn a Python function into a TensorFlow computation graph.
@tf.function
def debiasing_train_step(x, y):
  
  with tf.GradientTape() as tape:
    y_logit, z_mean, z_logsigma, x_recon = dbvae(x)
    loss, class_loss = debiasing_loss_function(x, x_recon, y, y_logit, z_mean, z_logsigma)
  
  grads = tape.gradient(loss, dbvae.trainable_variables)
  optimizer.apply_gradients(zip(grads, dbvae.trainable_variables))
  
  return loss

all_faces = loader.get_all_train_faces()
if hasattr(tqdm, '_instances'): tqdm._instances.clear()
  
for i in range(num_epochs):
  
  IPython.display.clear_output(wait=True)
  print("Starting epoch{}/{}".format(i+1, num_epochs))
  p_faces = get_training_sample_probabilities(ll_faces, dbvae)
  for j in tqdm(range(loader.get_train_size() // batch_size)):
    (x, y) = loader.get_batch(batch_size, p_pos=p_faces)
    loss = debiasing_train_step(x, y)
    
    if j % 500 == 0:
      mdl.utils.plot_sample(x, y, dbvae)
      
# Evaluation of DB-VAE on test dataset
dbvae_logits = [dbvae.predict(np.array(x, dtype=np.float32)) for x in test_faces]
dbvae_probs = tf.squeeze(tf.sigmoid(dbvae_logits))

xx = np.arange(len(keys))
plt.bar(xx, standard_classifier_probs.numpy().mean(1), width=0.2, label="Standard CNN")
plt.bar(xx+0.2, dbvae_probs.numpy().mean(1), width=0.2, label="DB-VAE")
plt.xticks(xx, keys); 
plt.title("Network predictions on test dataset")
plt.ylabel("Probability"); plt.legend(bbox_to_anchor=(1.04,1), loc="upper left");






