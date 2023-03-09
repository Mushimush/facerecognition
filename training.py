from keras.models import Model
from keras.layers import Input, Flatten, Dense
from sklearn.preprocessing import LabelBinarizer
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
import cv2
import glob
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# Create an ImageDataGenerator object with various augmentations
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=False,
                             fill_mode='nearest')


# Define the path to the directory containing the images
path = "/home/stanley/Documents/facerecognition/FR"

labels = ['jack', 'jk', 'stan']

# Load the images and labels into arrays
images = []
image_labels = []
for label in labels:
    image_paths = [os.path.join(path, label, f) for f in os.listdir(
        os.path.join(path, label)) if f.endswith('.jpg')]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        # Resize the image to 200 x 200 pixels
        img = cv2.resize(img, (224, 224))
        # Apply data augmentation
        img = datagen.random_transform(img)
        images.append(img)
        image_labels.append(label)

# Convert the image and label arrays to NumPy arrays
images = np.array(images)
image_labels = np.array(image_labels)

# Normalize the image array, the image vector values will be between 0 and 1. Convert to numpy array to perform mathematical expressions.

images = images/255

# Load the pre-trained VGGFace model
model = VGGFace(model='resnet50', include_top=False,
                input_shape=(224, 224, 3), pooling='avg')

label_binarizer = LabelBinarizer()
labels_one_hot = label_binarizer.fit_transform(image_labels)

# Freeze the pre-trained layers
for layer in model.layers:
    layer.trainable = False

# Add new layers on top of the pre-trained VGGFace model
input_layer = Input(shape=(224, 224, 3))
x = model(input_layer)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
output_layer = Dense(len(labels), activation='softmax')(x)
new_model = Model(inputs=input_layer, outputs=output_layer)

# Compile the new model
new_model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

# Train the new model
history = new_model.fit(images, labels_one_hot,
                        epochs=20, validation_split=0.2)
# Save the model
new_model.save('my_model.h5')
# Evaluate the new model
loss, accuracy = new_model.evaluate(images, labels_one_hot)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
