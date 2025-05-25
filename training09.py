import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, log_loss
import tensorflow as tf

DATASET_PATH = "datasets/09/09.csv"
dataset = np.loadtxt(open(DATASET_PATH, "rb"), delimiter=",")

x = dataset[:, :-1]
y = dataset[:, -1:]
x, y = shuffle(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
x_train /= 255
x_test /= 255
y_train_cat = tf.keras.utils.to_categorical(y_train)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(512, activation="relu", input_dim=x_train.shape[1]))
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train_cat, epochs=30, batch_size=512)

predict_train = model.predict(x_train)
predict_test = model.predict(x_test)
c_train = np.argmax(predict_train, axis=1)
c_test = np.argmax(predict_test, axis=1)

print(accuracy_score(y_train, c_train))
print(accuracy_score(y_test, c_test))
print(log_loss(y_train, predict_train))
print(log_loss(y_test, predict_test))

model.save("model09.h5")
