import cv2
import dlib
import os
import time

face_detector = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()

filename = 'test8.jpg'
model = 'haarcascade'
scale = '1'
raw_img = cv2.imread(os.path.join('TestImg', filename))
img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

t0 = time.time()
print('start')
faces = face_detector.detectMultiScale(img, 1.1, 1)
t1 = time.time()
print(f'took {round(t1-t0, 3)} to get {len(faces)} faces')

for (x, y, w, h) in faces:
    face = img[y:y + h, x:x + w][:, :, ::-1]
    cv2.rectangle(raw_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

font = cv2.FONT_HERSHEY_DUPLEX
text = f'took {round(t1-t0, 3)} to get {len(faces)} faces'
cv2.putText(raw_img, text, (20, 20), font, 0.5, (255, 255, 255), 1)
cv2.imwrite(os.path.join('TestOutput', f'{model}_{scale}_{filename}'), raw_img)

while True:
    cv2.imshow('IMG', raw_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break