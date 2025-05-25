import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import joblib

DATASET_PATH = "datasets/01/01.csv"
dataset = np.loadtxt(open(DATASET_PATH, "rb"), delimiter=",")

x = dataset[:, :-1]
y = dataset[:, -1:]
x, y = shuffle(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train /= 255
x_test /= 255

lr = LogisticRegression()
lr.fit(x_train, y_train)

predict_train = lr.predict(x_train)
proba_train = lr.predict_proba(x_train)
predict_test = lr.predict(x_test)
proba_test = lr.predict_proba(x_test)

print(accuracy_score(y_train, predict_train))
print(log_loss(y_train, proba_train))
print(accuracy_score(y_test, predict_test))
print(log_loss(y_test, proba_test))

joblib.dump(lr, "model01.pkl")
