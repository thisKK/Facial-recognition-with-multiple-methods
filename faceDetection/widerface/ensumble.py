import cv2
import numpy as np
import os
import time
from faceDetection import RetinaFace
from faceDetection.face_detection_DSFD import face_detection

# filename = 'test1366x768.jpg'
# raw_img = cv2.imread(os.path.join('../../testImage/', filename))
CONFIDENCE = 0.5

path = '../widerface/wider/WIDER_val/images'
out_file = '../widerface/wider/WIDER_prediction'

YoloConfig = '../face_detection_yolov3/Yolo/yolo_models/yolov3-face.cfg'
YoloWeight = '../face_detection_yolov3/Yolo/yolo_weights/yolov3-face.weights'
THRESH = 0.3
IMG_WIDTH, IMG_HEIGHT = 416, 416
net = cv2.dnn.readNetFromDarknet(YoloConfig, YoloWeight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def retinaModel(raw_img):
    detector = RetinaFace(gpu_id=0)
    faces = detector(raw_img)
    boxes = []
    for box, landmarks, score in faces:
        score = score.astype(np.float)
        box = box.astype(np.int)
        if score < CONFIDENCE:
            continue
        b = (box, score)
        boxes.append(b)
    return boxes

def dsfdModel(raw_img):
    detector = face_detection.build_detector(
        "DSFDDetector",
        max_resolution=1080
    )
    dets = detector.detect(raw_img[:, :, ::-1])[:, :5]
    boxes = []
    for box in dets:
        if box[4].astype(float) < CONFIDENCE:
            continue
        bbox = box[:4].astype(int)
        b = (bbox, box[4])
        boxes.append(b)
    return boxes

def yoloModel(raw_img):
    h, w, _ = raw_img.shape
    blob = cv2.dnn.blobFromImage(raw_img, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)

    net.setInput(blob)
    layers_names = net.getLayerNames()
    outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])

    boxes = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # only face

            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([w, h, w, h])
                centerX, centerY, bwidth, bheight = box.astype('int')
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))

        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESH)
        box = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                # x, y = boxes[i][0], boxes[i][1]
                # w, h = boxes[i][0]+boxes[i][2], boxes[i][1]+boxes[i][3]
                bbox = [boxes[i][0], boxes[i][1], boxes[i][0]+boxes[i][2], boxes[i][1]+boxes[i][3]]
                box.append((np.array(bbox), confidences[i]))
    return box

if __name__ == "__main__":
    for dir in os.listdir(path):
        folder_name = dir
        os.mkdir(os.path.join(out_file, folder_name))
        print(folder_name)

        for fn in os.listdir(os.path.join(path, folder_name)):
            boxes = []
            confidences = []
            raw_img = cv2.imread(os.path.join(path, folder_name, fn))
            reatina = retinaModel(raw_img)
            for face in reatina:
                boxes.append(face[0])
                confidences.append(float(face[1]))
                # cv2.rectangle(raw_img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)
            dsfd = dsfdModel(raw_img)
            for face in dsfd:
                boxes.append(face[0])
                confidences.append(float(face[1]))
                # cv2.rectangle(raw_img, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=1)
            yolo = yoloModel(raw_img)
            for face in yolo:
                boxes.append(face[0])
                confidences.append(float(face[1]))
                # cv2.rectangle(raw_img, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=1)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, 0.85)
            print(len(idxs))

            prediction_file = os.path.join(out_file, folder_name, fn.replace('jpg', 'txt'))
            name = fn.split('.')
            name = name[0]

            with open(prediction_file, 'w') as f:
                f.write("%s\n" % str(name))
                f.write("%d\n" % len(idxs))
            if len(idxs) > 0:
                for i in idxs.flatten():
                    x, y = boxes[i][0], boxes[i][1]
                    w, h = boxes[i][2], boxes[i][3]
                    # cv2.rectangle(raw_img, (x, y), (w, h), color=(255, 255, 255), thickness=1)
                    with open(prediction_file, 'a') as f:
                        f.write("%d %d %d %d %g\n" % (x, y, w, h, confidences[i]))
            print("Process Image" + " " + name)

    while True:
        cv2.imshow('IMG', raw_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

