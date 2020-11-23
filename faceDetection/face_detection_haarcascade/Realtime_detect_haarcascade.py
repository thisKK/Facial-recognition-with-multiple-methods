import cv2
import dlib

face_detector = cv2.CascadeClassifier('../haarcascade/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture('../TestVideo/maskoff.mp4')
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 3)  # 1.05 is less as posible
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()