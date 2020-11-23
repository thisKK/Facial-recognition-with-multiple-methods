import cv2
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
    childPath.text = outputPath + "/" + filename
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



# filename = 'obama.jpg'
model = 'yolov3-face'
path = '../data/29--Students_Schoolkids/'
scale = 1
IMG_WIDTH, IMG_HEIGHT = 416, 416
CONFIDENCE = 0.5
THRESH = 0.3
LABELS = 'face'
output_filepath = '../data'

net = cv2.dnn.readNetFromDarknet("../Yolo/yolo_models/yolov3-face.cfg", "../Yolo/yolo_weights/yolov3-face.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


for fn in os.listdir(path):
    filename = fn
    raw_img = cv2.imread(os.path.join(path, filename))
    out_file = '../data'
    h, w, _ = raw_img.shape
    wI, hI, d = raw_img.shape
    name = fn.split('.')
    name = name[0]
    out_file = os.path.join(out_file, name.replace('jpg', 'xml'))
    count = 0
    # inference
    t0 = time.time()

    blob = cv2.dnn.blobFromImage(raw_img, 1/255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layers_names = net.getLayerNames()
    outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])

    boxes = []
    confidences = []
    class_ids = []
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

    t1 = time.time()
    print(f"took {round(t1-t0, 3)} to get {len(idxs.flatten())} faces")
    boxes1 = []

    if boxes is None or confidences is None or idxs is None or class_ids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'

    for i in idxs.flatten():
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        box = x, y, w+x, h+y
        box = np.asarray(box, dtype=np.float32)
        if confidences[i] < CONFIDENCE:
            continue
        boxes1.append(([LABELS, box], confidences[i]))
        with open(out_file + '.xml', 'w') as f:
            f.write(generateXML(filename, output_filepath, hI, wI, d, boxes1))

    # while True:
    #     cv2.imshow('IMG', raw_img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
