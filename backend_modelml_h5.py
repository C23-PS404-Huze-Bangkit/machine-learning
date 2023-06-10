# File untuk dikonversi menjadi model backend pengenalan gambar untuk dilakukan menggunakan flask dan dijalankan di cloudrun
# diperlukan pip install tensorflow and numpy
#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

classes = (
    "American Shorthair",
    "Basset hound",
    "Beagle",
    "Bengal",
    "Boxer",
    "British Shorthair",
    "Chihuahua",
    "English cocker spaniel",
    "Japanese chin",
    "Maine Coon",
    "Newfoundland",
    "Persian",
    "Pomeranian",
    "Pug",
    "Ragdoll",
    "Russian Blue",
    "Samoyed",
    "Scottish fold",
    "Siamese",
    "Sphynx"
)

model = tf.keras.models.load_model('Xception_datafix.h5')

img_path = 'Untitled.jpg'
img = image.load_img(img_path, target_size=(299, 299))
input_data = image.img_to_array(img)
input_data = (input_data / 255.0)
input_data = np.expand_dims(input_data, axis=0)

predictions = model.predict(input_data)

predicted_class_index = np.argmax(predictions[0])
predicted_class = classes[predicted_class_index]
confidence = predictions[0][predicted_class_index] * 100

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")