from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileItem(BaseModel):
    name : str

# Import the required libraries for the project
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import *
import keras.utils as image
from keras.preprocessing.image import ImageDataGenerator
import os, shutil
import warnings
warnings.filterwarnings('ignore')


# Let's plot a few images
train_path = "D:/DATASET/train"
validation_path = "D:/DATASET/validation"
test_path = "D:/DATASET/test"

image_categories = os.listdir('D:/DATASET/train')

def plot_images(image_categories):
    
    # Create a figure
    plt.figure(figsize=(12, 12))
    for i, cat in enumerate(image_categories):
        
        # Load images for the ith category
        image_path = train_path + '/' + cat
        images_in_folder = os.listdir(image_path)
        first_image_of_folder = images_in_folder[0]
        first_image_path = image_path + '/' + first_image_of_folder
        img = image.load_img(first_image_path)
        img_arr = image.img_to_array(img)/255.0
        
        
        # Create Subplot and plot the images
        plt.subplot(1, 3, i+1)
        plt.imshow(img_arr)
        plt.title(cat)
        plt.axis('off')
        
    
# Creating Image Data Generator for train, validation and test set

# 1. Train Set
train_gen = ImageDataGenerator(rescale = 1.0/255.0,rotation_range=30) # Normalise the data
train_image_generator = train_gen.flow_from_directory(
                                            train_path,
                                            target_size=(150, 150),
                                            batch_size=32,
                                            class_mode='categorical')

# 2. Validation Set
val_gen = ImageDataGenerator(rescale = 1.0/255.0,rotation_range=30) # Normalise the data
val_image_generator = train_gen.flow_from_directory(
                                            validation_path,
                                            target_size=(150, 150),
                                            batch_size=32,
                                            class_mode='categorical')

# 3. Test Set
test_gen = ImageDataGenerator(rescale = 1.0/255.0,rotation_range=30) # Normalise the data
test_image_generator = train_gen.flow_from_directory(
                                            test_path,
                                            target_size=(150, 150),
                                            batch_size=32,
                                            class_mode='categorical')

# Print the class encodings done by the generators
class_map = dict([(v, k) for k, v in train_image_generator.class_indices.items()])

# Build a custom sequential CNN model

model = Sequential() # model object

# Add Layers
model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=[150, 150, 3]))
model.add(MaxPooling2D(2, ))

model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(2))

model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(2))

model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(2))

model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D(2))

# Flatten the feature map
model.add(Flatten())

# Add the fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dense(3, activation='softmax'))


early_stopping = keras.callbacks.EarlyStopping(patience=5)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy','Precision','Recall'])
hist = model.fit(train_image_generator, 
                 epochs=20, 
                 verbose=1, 
                 validation_data=val_image_generator, 
                 steps_per_epoch = 487//32, 
                 validation_steps = 158//32, 
                 callbacks=early_stopping)

 # Define the variables for matplotlib settings
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

train_precision = hist.history['precision']
val_precision = hist.history['val_precision']

train_recall = hist.history['recall']
val_recall = hist.history['val_recall']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(len(acc))

# Accuracy chart
plt.plot(epochs, acc, 'r', label ='Training accuracy')
plt.plot(epochs, val_acc, 'b', label ='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

# Precision chart
plt.plot(epochs, train_precision, 'r', label = 'Training precision')
plt.plot(epochs, val_precision, 'b', label = 'Validation precision')
plt.title('Training and validation precision')
plt.legend()
plt.figure()

# Recall chart
plt.plot(epochs, train_recall, 'r', label = 'Training recall')
plt.plot(epochs, val_recall, 'b', label = 'Validation recall')
plt.title('Training and validation recall')
plt.legend()
plt.figure()

# Loss function chart
plt.plot(epochs, loss, 'r', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and validation loss')
plt.legend()

# Predict the accuracy for the test set
model.evaluate(test_image_generator)

# We test our Simple CNN model to get the metrics scores, 3000 is the size of the images we have in the test dataset 
eval_result1 = model.evaluate_generator(test_image_generator, 149)

@app.post('/')
async def scoring_endpoint(item:FileItem):

# Testing the Model
    test_image_path = item.name
#def generate_predictions(test_image_path, actual_label):
    
    # 1. Load and preprocess the image
    test_img = image.load_img(test_image_path, target_size=(150, 150))
    test_img_arr = image.img_to_array(test_img)/255.0
    test_img_input = test_img_arr.reshape((1, test_img_arr.shape[0], test_img_arr.shape[1], test_img_arr.shape[2]))

    # 2. Make Predictions
    predicted_label = np.argmax(model.predict(test_img_input))
    predicted_candy = class_map[predicted_label]
    return predicted_candy





