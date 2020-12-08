import cv2
from cv2 import dnn
import os
import time
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def generateXML(filename,outputPath,w,h,d,boxes):
    top = ET.Element('annotation')
    childFolder = ET.SubElement(top, 'folder')
    childFolder.text = 'images'
    childFilename = ET.SubElement(top, 'filename')
    childFilename.text = filename[0:filename.rfind(".")]
    childPath = ET.SubElement(top, 'path')
    childPath.text = outputPath
    childSource = ET.SubElement(top, 'source')
    childDatabase = ET.SubElement(childSource, 'database')
    childDatabase.text = 'Unknown'
    childSize = ET.SubElement(top, 'size')
    childWidth = ET.SubElement(childSize, 'width')
    childWidth.text = str(w)
    childHeight = ET.SubElement(childSize, 'height')
    childHeight.text = str(h)
    childDepth = ET.SubElement(childSize, 'depth')
    childDepth.text = str(d)
    childSegmented = ET.SubElement(top, 'segmented')
    childSegmented.text = str(0)
    for (box,score) in boxes:
        category = box[0]
        box = box[1].astype("int")
        (x,y,xmax,ymax) = box
        childObject = ET.SubElement(top, 'object')
        childName = ET.SubElement(childObject, 'name')
        childName.text = category
        childScore = ET.SubElement(childObject, 'confidence')
        childScore.text = str(score)
        childPose = ET.SubElement(childObject, 'pose')
        childPose.text = 'Unspecified'
        childTruncated = ET.SubElement(childObject, 'truncated')
        childTruncated.text = '0'
        childDifficult = ET.SubElement(childObject, 'difficult')
        childDifficult.text = '0'
        childBndBox = ET.SubElement(childObject, 'bndbox')
        childXmin = ET.SubElement(childBndBox, 'xmin')
        childXmin.text = str(x)
        childYmin = ET.SubElement(childBndBox, 'ymin')
        childYmin.text = str(y)
        childXmax = ET.SubElement(childBndBox, 'xmax')
        childXmax.text = str(xmax)
        childYmax = ET.SubElement(childBndBox, 'ymax')
        childYmax.text = str(ymax)
    return prettify(top)


model = 'Yolov3'

path = '../widerface/wider/WIDER_val/images'
out_file = '../widerface/wider/WIDER_prediction'

YoloConfig = '../face_detection_yolov3/Yolo/yolo_models/yolov3-face.cfg'
YoloWeight = '../face_detection_yolov3/Yolo/yolo_weights/yolov3-face.weights'

count = 0
CONFIDENCE = 0.5
THRESH = 0.3
IMG_WIDTH, IMG_HEIGHT = 416, 416
LABELS = 'face'

net = cv2.dnn.readNetFromDarknet(YoloConfig, YoloWeight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
if __name__ == "__main__":
    for dir in os.listdir(path):
        folder_name = dir
        os.mkdir(os.path.join(out_file, folder_name))
        print(folder_name)

        for fn in os.listdir(os.path.join(path, folder_name)):
            raw_img = cv2.imread(os.path.join(path, folder_name, fn))
            h, w, d = raw_img.shape
            blob = cv2.dnn.blobFromImage(raw_img, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            layers_names = net.getLayerNames()
            outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])

            prediction_file = os.path.join(out_file, folder_name, fn.replace('jpg', 'xml'))

            blob = cv2.dnn.blobFromImage(raw_img, 1/255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            layers_names = net.getLayerNames()
            outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])

            boxes = []
            confidences = []
            class_ids = []
            box2 = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    # only face
                    if confidence > CONFIDENCE and class_id == 0:
                        box = detection[0:4] * np.array([w, h, w, h])
                        centerX, centerY, bwidth, bheight = box.astype('int')
                        x = int(centerX - (bwidth / 2))
                        y = int(centerY - (bheight / 2))
                        boxes.append([x, y, int(bwidth), int(bheight)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESH)
            boxes1 = []

            if boxes is None or confidences is None or idxs is None or class_ids is None:
                raise '[ERROR] Required variables are set to None before drawing boxes on images.'
            if len(idxs) > 0:
                for i in idxs.flatten():
                    x, y = boxes[i][0], boxes[i][1]
                    w, h = boxes[i][2], boxes[i][3]
                    box = x, y, w+x, h+y
                    box = np.asarray(box, dtype=np.float32)
                    if confidences[i] < CONFIDENCE:
                        continue
                    boxes1.append(([LABELS, box], confidences[i]))
                    with open(prediction_file, 'w') as f:
                        f.write(generateXML(fn, os.path.join(path, folder_name, fn), h, w, d, boxes1))
            print("Process Image" + " " + fn)

            # while True:
            #     cv2.imshow('IMG', raw_img)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
