import time
import os

from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import argparse
import json
# Create  praser
parser = argparse.ArgumentParser(description='Command-line application (py script) to predict the type of given flower using pre-trained model')

# Parser arguments
# Positional arguments 
parser.add_argument("image_path", help="path to the input image folder", type=str)
parser.add_argument("Final_model", help="path to the Final saved  model", type=str)

# Optional arguments 
parser.add_argument("-k", "--top_k", default=5, help ="top k class probabilities", type=int)
parser.add_argument("-n", "--category_names", default="./label_map.json", help="path to a JSON file mapping labels to the actual flower names", type=str)

args = parser.parse_args()  

image_path = args.image_path
Final_model = args.Final_model
top_k = args.top_k
category_names = args.category_names


# Load a JSON file that maps the class values to other category names
with open(category_names, 'r') as f:
    class_names = json.load(f)


# Load the Keras model
reloaded_model = tf.keras.models.load_model(Final_model , custom_objects={'KerasLayer':hub.KerasLayer})
print(reloaded_model.summary())

# Create the process_image function

def process_image(image):
    print(f'Original image shape is {image.shape}')
    image = tf.cast(image , tf.float32)
    image = tf.image.resize(image , (224 , 224))
    image /= 255
    return image
# Predict the top K flower classes along with associated probabilities
def predict(path , model , top_k):
	
    #Prepare images
    image = Image.open(path)
    image = np.asarray(image)
    processed_image = np.expand_dims(process_image(image) , axis=0)

    model_preds = model.predict(processed_image)[0].tolist()
    top_k_probs , top_k_classes = tf.math.top_k(model_preds , k = top_k)
    top_k_probs = top_k_probs.numpy().tolist()
    top_k_classes = top_k_classes.numpy().tolist()
  
    return top_k_probs , top_k_classes ,processed_image
    
probs, classes , processed_image = predict(image_path, reloaded_model, top_k)
flower_name=flower = image_path.split('/' )[-1].split('.')[0]
class_names_pred = []
print(f'The top {len(classes)} classes and their probabilites for {flower_name} is : ')
for i in range(len(classes)):
    class_names_pred.append(class_names[str(classes[i]+1)])
    print(class_names_pred[i], ':', probs[i])
