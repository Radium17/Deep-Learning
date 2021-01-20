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

# MNIST dataset

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
train_images = (np.expand_dims(train_images, axis=-1)/255.).astype(np.float32)
train_labels = (train_labels).astype(np.int64)
test_images = (np.expand_dims(test_images, axis=-1)/255.).astype(np.float32)
train_labels = (test_labels).astype(np.int64)

# Visualization

plt.figure(figsize=(10,10))
random_inds = np.random.choice(60000,36)
for i in range(36):
 plt.subplot(6,6,i+1)
 plt.xticks([])
 plt.yticks([])
 plt.grid(False)
 image_ind = random_inds[i]
 plt.imshow(np.squeeze(train_images[image_ind]), cmap=plt.cm.binary)
 plt.xlabel(train_labels([image_ind]))
 
 # Fully connected NN architecture for Handwritten Digit Classification
 
 def build_fc_model():
  fc_model = tf.keras.Sequential([
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(128, activation = 'relu')
   tf.keras.layers.Dense(10, activation = 'softmax')   
  ])
  return fc_model
 
 model = build_fc_model()
 
 model.compile(optimizer = tf.keras.optimizers.Adam(learnin_rate=1e-3),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

 # Train the model
 
 BATCH_SIZE = 64
 EPOCHS = 5
 
 model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)
 
 # Evaluate on the test dataset
 
 test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)
 
 print('Test accuracy:', test_acc)
 
 
 # CNN model
 
 def build_cnn_model():
  cnn_model = tf.keras.Sequential([
   
   tf.keras.layers.Conv2D(24,3),
   tf.keras.layers.MaxPool2D(pool_size=(2,2)),
   tf.keras.layers.Conv2D(36,3),
   tf.keras.layers.MaxPool2D(pool_size=(2,2)),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dense(10, activation='softmax')
  ])
  return cnn_model

cnn_model = build_cnn_model()
cnn_model.predict(train_images[[0]])
print(cnn_model.summary())

# Train the CNN model

cnn_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Evaluate the CNN model

test_loss, test_acc = cnn_model.evaluate(test_images, test_labels, BATCH_SIZE)

print('Test accuracy', test_acc)

# Make predictions

predictions = cnn_model.predict(test_images)
print(predictions[0])
prediction  = np.argmax(predictions[0])
print(prediction)

print("Label of this digit is:", test_labels[0])
plt.imshow(test_images[0,:,:,0], cmap=plt.cm.binary)

# To visualize with a slider

#@title Change the slider to look at the model's predictions! {run: "auto"}

image_index = 2 #@param {type:"slider", min:0, max:100, step:1}
plt.subplot(1,2,1)
mdl.lab2.plot_image_prediction(image_index, predictions, test_labels, test_images)
plt.subplot(1,2,2)
mdl.lab2.plot_value_prediction(image_index, predictions,  test_labels)


# Plots the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red

num_rows = 5
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  mdl.lab2.plot_image_prediction(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  mdl.lab2.plot_value_prediction(i, predictions, test_labels)

 # Rebuild and train the CNN model

cnn_model = build_cnn_model()

batch_size = 12
loss_history = mdl.util.LossHistory(smoothing_factor=0.95) # to record the evolution of the loss
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss', scale='semilogy')
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2) # define our optimizer

if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

for idx in tqdm(range(0, train_images.shape[0], batch_size)):
  # First grab a batch of training data and convert the input images to tensors
  (images, labels) = (train_images[idx:idx+batch_size], train_labels[idx:idx+batch_size])
  images = tf.convert_to_tensor(images, dtype=tf.float32)

  # GradientTape to record differentiation operations
  with tf.GradientTape() as tape:
    logits = cnn_model(images)
    loss_value = tf.keras.backend.sparse_categorical_crossentropy(labels, logits) # TODO

  loss_history.append(loss_value.numpy().mean()) # append the loss to the loss_history record
  plotter.plot(loss_history.get())

  # Backpropagation
  # cnn_model.trainable_variables to access these parameters 
  grads = tape.gradient(loss_value, cnn_model.trainable_variables)
  optimizer.apply_gradients(zip(grads, cnn_model.trainable_variables))

 
 
