import cv2
import sys
from datetime import datetime

def rectangle_contains(r1, r2):
    return r1[0] < r2[0] < r2[0] + r2[2] < r1[0] + r1[2] and r1[1] < r2[1] < r2[1] + r2[3] < r1[1] + r1[3]

rec = False
mode_1 = False
mode_2 = False

video_capture = cv2.VideoCapture(0)
codec = cv2.VideoWriter_fourcc(*'MJPG')
out = None

b, _ = video_capture.read()

if not b:
    print('Error')
    sys.exit()

haarcascade_frontalface_default = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
haarcascade_smile = cv2.CascadeClassifier("haarcascade_smile.xml")

while video_capture.isOpened():
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode_1:
        frame = gray
    if mode_2:
        now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        cv2.putText(frame, now, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    if rec:
        out.write(frame)
        cv2.circle(frame, (frame.shape[1] - 30, frame.shape[0] - 30), 10, (0, 0, 255), cv2.FILLED)

    faces = haarcascade_frontalface_default.detectMultiScale(gray, 1.1, 10)
    smiles = haarcascade_smile.detectMultiScale(gray, 1.1, 30)
    for f in faces:
        cv2.rectangle(frame, (f[0], f[1]), (f[0] + f[2], f[1] + f[3]), (0, 255, 0), 2)
        for s in smiles:
            if rectangle_contains(f, s):
                cv2.rectangle(frame, (s[0], s[1]), (s[0] + s[2], s[1] + s[3]), (255, 0, 0), 2)
                face = frame[f[1]:f[1]+f[3], f[0]:f[0]+f[2]]
                cv2.imwrite('face.jpg', face)

    cv2.imshow('Video', frame)
    k = cv2.waitKey(1)

    if k == ord('q'):
        break
    elif k == ord('1'):
        mode_1 = not mode_1
    elif k == ord('2'):
        mode_2 = not mode_2
    elif k == ord('3'):
        name = datetime.now().strftime('%Y%m%d%H%M%S%f') + '.jpg'
        cv2.imwrite(name, frame)
    elif k == ord('4'):
        if out == None:
            out = cv2.VideoWriter('output.avi', codec, 20, (640, 480))
        rec = not rec

if out != None:
    out.release()
video_capture.release()
cv2.destroyAllWindows()
