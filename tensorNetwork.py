"""
	Program: Neural Network Deep Learning with Tensor Flow
	Author: Daro Omwanor
	Date: August 6 2019
"""

import os 
import tensorflow as tf 
from skimage import data, transform
from skimage.color import rgb2gray
import matplotlib 
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import random

class tensorNetwork:

	"""
		Optimization algorithms that can be used are the Stochastic Gradient Descent (SGD), ADAM and RMSprop.
		ADAM optimizer algorithm for which we define the learning rate at 0.001 is used here.
	"""
	def __init__(self,images,labels):

#Initialize placeholders
		x = tf.placeholder(dtype = tf.float32, shape = [None,28, 28])
		y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data

		images_flat = tf.contrib.layers.flatten(x)

# Fully  Connected Layer

		logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function

		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

# Define an optimizer

		train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes

		correct_pred = tf.argmax(logits, 1)

# Defibe an accuracy metric

		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		
#Begin tensorflow Session

		tf.set_random_seed(1234)

		sess = tf.Session()


		sess.run(tf.global_variables_initializer())



		images28 = [transform.resize(image, (28, 28)) for image in images]
		images28 = np.array(images28)
		images28 = rgb2gray(images28)

		for i in range(201):
			print('EPOCH', i)
			_, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})

			if i % 10 == 0:
				print("Loss: ", loss)
			print('DONE WITH EPOCH')

		sample_indexes = random.sample(range(len(images28)), 10)

		sample_images = [images28[i] for i in sample_indexes]

		sample_labels = [labels[i] for i in sample_indexes]

		predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

		print(sample_labels)

		print(predicted)

		fig = plt.figure(figsize=(10, 10))

		for i in range(len(sample_images)):
			truth = sample_labels[i]
			prediction = predicted[i]
			plt.subplot(5, 2,1+i)
			plt.axis('off')
			color='green' if truth == prediction else 'red'
			plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), fontsize=12, color=color) 
			plt.imshow(sample_images[i],  cmap="gray")
		plt.show()
		sess.close()

def load_data(data_directory):
	directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
	labels = []
	images = []
	for d in directories:
		label_directory = os.path.join(data_directory, d)
		file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(".ppm")]
		for f in file_names:
			images.append(data.imread(f))
			labels.append(int(d))
	return images,labels



if __name__ == '__main__':

	
	ROOT_PATH = "/Users/omwanord"
	train_data_directory = os.path.join(ROOT_PATH, "pyTest/img_data/Training")
	test_data_directory = os.path.join(ROOT_PATH, "pyTest/img_data/Testing")
	images, labels = load_data(train_data_directory)
	node = tensorNetwork(images,labels)
	
	