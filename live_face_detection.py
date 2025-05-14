import cv2
import sys

video_capture = cv2.VideoCapture(1)
haarcascade_frontalface_default = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

b, _ = video_capture.read()
if not b:
    print('Error')
    sys.exit()

while video_capture.isOpened():
    _, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haarcascade_frontalface_default.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
