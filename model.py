# dependencies
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models

print("Version: ", tf.__version__)  # Check tf version
print("GPU is",
      "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")  # Check GPU status
physical_devices = tf.config.experimental.list_physical_devices('GPU')  # Config GPU
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Model:
    def __init__(self):
        pass

    # Image Processing
    def process(self, image):
        img = Image.fromarray(image)
        img_rescaled = img.resize((128, 128))
        img_array = np.asarray(img_rescaled)
        img_normalized = img_array / 255.0
        img_nparray = np.asarray(img_normalized).astype('float16')
        img_dim = np.expand_dims(img_nparray, axis=0)
        return img_dim


    # Model
    def create_model(self):
        model = models.load_model('model.h5')
        return model

    # Predictions
    def predictor(self, model, img):
        prediction = model.predict(img)
        #return np.argmax(prediction[0])
        return prediction[0][0], prediction[0][1]
