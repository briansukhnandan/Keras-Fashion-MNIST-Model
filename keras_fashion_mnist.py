import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Disables AVX2 and FMA warnings on Unix-based systems.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dataset built into Keras. See https://keras.io/datasets/
# Each picture is an array of 28x28 pictures.
data = keras.datasets.fashion_mnist

# We split our dataset into 2 sections, ~70-80% for training, then the remaining
# amount to test for accuracy once the model has been trained.
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Defines the index that will be returned by the model when given a piece of data.
# For example, returning 9, corresponds to class_names[9] which is an Ankle boot.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Outputs arrays of pixel values, ranging from 0-255.
# For our testing purposes, we want them to be out of 0-1, so we divide each value by 255.
# NumPy lets us do this.
# Think of this as shrinking down our data.
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.imshow() shows the image we specify from the dataset. plt.cm.binary shows in black/white.
# plt.imshow(train_images[0], cmap=plt.cm.binary)
# plt.show()

# The thing to understand is we take our 2d array and flatten it into a bigass 1d array.
# Since each pic is 28x28 pixels, we flatten it down to a 1d array of size 784,

# Defines our Sequential model:
model = keras.Sequential([

    # Create a first layer of 784 neurons (flatten 28x28 2d array.
    keras.layers.Flatten(input_shape=(28,28)),

    # Create a dense layer of 128 neurons, a dense layer is a fully connected layer.
    # The activation function will be the recitifed linear unit function.
    keras.layers.Dense(128, activation="relu"),

    # Makes sure that our last layer's neurons add up to 1, so that we can take the max.
    keras.layers.Dense(10, activation="softmax")

])

# Sets up parameters for our model. The loss function is the sparse categorical crossentropy function,
# and with metrics we are looking for accuracy, essentially how low we can get the loss function to be.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Epochs tell us how many times the model will see the information. Basically how many times we see an image.
model.fit(train_images, train_labels, epochs=8)

prediction = model.predict(test_images)

# Set 2 variables to be our results from loss function, and tested accuracy.
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("\nAccuracy of Model:", test_acc*100, "%")

# This will output what our prediction is based on what the model thinks.
# Basically every neuron in our last layer (10 neurons) will have a value between 0-1.
# We will take the max of these.
for i in range(len(test_labels)-1):
    plt.grid(False)

    # Display image to plot.
    plt.imshow(test_images[i], cmap=plt.cm.binary)

    # Print actual classification on the x-axis.
    plt.xlabel("Actual classification: " + class_names[test_labels[i]])

    # Take the max of the prediction array, which will be size 10 (The amount of output neurons.
    plt.title("Prediction from Model: " + class_names[np.argmax(prediction[i])])
    plt.show()

