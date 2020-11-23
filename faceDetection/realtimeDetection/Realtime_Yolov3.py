import cv2
from utils import *

model = 'yolov3-face'
scale = 1

IMG_WIDTH, IMG_HEIGHT = 416, 416
CONFIDENCE = 0.2
THRESH = 0.3

net = cv2.dnn.readNetFromDarknet("../Yolo/yolo_models/yolov3-face.cfg", "../Yolo/yolo_weights/yolov3-face.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
cap = cv2.VideoCapture('../TestVideo/maskon.mp4')
while True:

        has_frame, frame = cap.read()

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            cv2.waitKey(1000)
            break

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        print('[i] ==> # detected faces: {}'.format(len(faces)))
        print('#' * 60)

        # initialize the set of information we'll displaying on the frame
        info = [
            ('number of faces detected', '{}'.format(len(faces)))
        ]
        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            # fps = cap.get(cv2.CAP_PROP_FPS)
            # cv2.putText(frame, str(cap.get(cv2.CAP_PROP_FPS)), (200,100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        wind_name = 'face detection using YOLOv3'
        cv2.imshow(wind_name, frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break
q
print('==> All done!')
print('***********************************************************')
cap.release()
cv2.destroyAllWindows()
