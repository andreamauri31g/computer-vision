import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

DATASET_PATH = "datasets/catdog/catdog.csv"
dataset = np.loadtxt(open(DATASET_PATH, "rb"), delimiter=",")

x = dataset[:, :-1]
y = dataset[:, -1:]
x, y = shuffle(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train /= 255
x_test /= 255

x_train = x_train.reshape((x_train.shape[0], 64, 64, 1))
x_test = x_test.reshape((x_test.shape[0], 64, 64, 1))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding="same", activation="relu", input_shape=(64, 64, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=30, batch_size=128)

predict_train = model.evaluate(x_train, y_train)
predict_test = model.evaluate(x_test, y_test)

print(f"Train -> {predict_train}")
print(f"Test -> {predict_test}")

model.save("modelcatdog.h5")
