import cv2
import numpy as np
import os
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


colors = np.random.randint(0, 255, size=(10, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]
rank_model = tf.keras.models.load_model("rank_classification_model.h5")
suit_model = tf.keras.models.load_model("suit_classification_model.h5")
rank_img_size = (70, 125)
suit_img_size = (70, 100)
