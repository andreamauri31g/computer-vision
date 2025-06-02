import tensorflow as tf
import os
import cv2
import numpy as np

DATASET_PATH = "datasets/"

p = []

for f in os.listdir(DATASET_PATH):
    if(".jpg" in f):
        img = cv2.imread(DATASET_PATH + f)
        i = cv2.resize(img, (100, 100))
