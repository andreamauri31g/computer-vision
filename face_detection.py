import cv2

img = cv2.imread("budTerence_set2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

haarcascade_frontalface_default = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces = haarcascade_frontalface_default.detectMultiScale(gray, 1.1, 6)

for (x, y, w, h) in faces:
    cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 2)

cv2.imshow('gray', gray)
cv2.waitKey(0)

cv2.destroyAllWindows()
