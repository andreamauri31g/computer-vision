import cv2
import numpy as np
import tensorflow as tf

SIZE = (28, 28)

np.set_printoptions(precision=3, floatmode='fixed')

model = tf.keras.models.load_model("model09.h5")

img_path = input("Img path: ")
im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(im, SIZE)
x = img.flatten().astype(float)
x /= 255.0
x = 1.0 - x
x = x.reshape(1, x.shape[0])

predict = model.predict(x)
c = np.argmax(predict, axis=1)

print(np.round(predict[0], 3))
print(c[0])
