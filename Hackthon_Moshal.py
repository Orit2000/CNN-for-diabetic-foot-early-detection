import numpy as np  # math functions
import scipy  # scientific functions
import matplotlib.pyplot as plt  # for plotting figures and setting their properties
import pandas as pd  # handling data structures (loaded from files)
from scipy.stats import linregress  # contains linregress (for linear regression)
from scipy.optimize import curve_fit as cfit  # non-linear curve fitting
from sklearn.metrics import r2_score  # import function that calculates R^2 score
from prettytable import PrettyTable  # display data in visually table format
import scipy.ndimage as ndimage  # import Multidimensional Image processing package
from scipy.signal import find_peaks  # for find local peaks in a signal
import os
import tensorflow

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories for your dataset
train_dir = os.path.join(os.getcwd(), 'Data')   # with both bad and good samples

# Create an ImageDataGenerator for preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,  # Scale pixel values to the range [0, 1]
    validation_split=0.1  # Split data into training and validation sets
)

# Load and preprocess the training data
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(700, 350),  # Resize images to a common size
    batch_size=32,  # Batch size for training
    class_mode='binary',  # Set this to 'binary' for binary classification
    subset='training'  # Specify this as 'training' for training data
)

# Load and preprocess the validation data
validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(700, 350),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Specify this as 'validation' for validation data
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create a Sequential model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(700, 350, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Flatten the feature maps
model.add(Flatten())

# Add a fully connected layer
model.add(Dense(128, activation='relu'))

# Add the output layer for binary classification (1 neuron, sigmoid activation)
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10  # Adjust the number of epochs as needed
)

loss, accuracy = model.evaluate(validation_generator)
print(f'Validation loss: {loss:.2f}')
print(f'Validation accuracy: {accuracy:.2%}')

# Load and preprocess a new image (replace 'path/to/your/image.jpg' with the actual image path)
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = os.path.join(os.getcwd(), 'Test', 'test2_unhealthy.png')   # with both bad and good samples
img = image.load_img(img_path, target_size=(700, 350))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize pixel values

# Make a prediction
prediction = model.predict(img_array)

if prediction > 0.5:
    print("Unhealthy Foot")
else:
    print("Healthy Foot")

print("Finished.:)")

'''
import cv2
import os

dimensions=[]
# Directory containing the PNG files
directory_path = os.path.join(os.getcwd(), 'deleteme')
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    img=cv2.imread(file_path)
    # Get the width and height
    height= img.shape[0]
    width = img.shape[1]
    dimensions.append((width, height))

# Sort the list of dimensions by width
dimensions.sort()

# Calculate the median width and height
median_index = len(dimensions) // 2

median_width, median_height = dimensions[median_index]

print(f"Median Width: {median_width}")
print(f"Median Height: {median_height}")
'''