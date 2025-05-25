import cv2
import joblib

SIZE = (28, 28)

lr = joblib.load("model.pkl")

img_path = input("Img path: ")
im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(im, SIZE)
x = img.flatten().astype(float)
x /= 255.0
x = 1.0 - x
x = x.reshape(1, x.shape[0])

predict = lr.predict(x)
proba = lr.predict_proba(x)

print(predict[0])
print(proba[0])
