import cv2
import os
import time
import glob
from faceDetection.face_detection_DSFD import face_detection
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

if __name__ == "__main__":
    LABELS = 'face'
    path = '../widerface/wider/WIDER_val/images'
    out_file = '../widerface/wider/WIDER_prediction'
    impaths = glob.glob(os.path.join(path, "*.jpg"))
    CONFIDENCE = 0.5
    detector = face_detection.build_detector(
        "DSFDDetector",
        max_resolution=1080
    )
    for dir in os.listdir(path):
        folder_name = dir
        os.mkdir(os.path.join(out_file, folder_name))
        print(folder_name)

        for fn in os.listdir(os.path.join(path, dir)):
            raw_img = cv2.imread(os.path.join(path, folder_name, fn))
            hI, wI, d = raw_img.shape

            t0 = time.time()
            dets = detector.detect(raw_img[:, :, ::-1])[:, :5]
            t1 = time.time()
            prediction_file = os.path.join(out_file, dir, fn.replace('jpg', 'xml'))
            name = fn.split('.')
            name = name[0]
            box2 = []
            boxes1 = []
            for box in dets:
                if box[4].astype(float) < CONFIDENCE:
                    continue
                bbox = box[:4].astype(int)
                b = (bbox, box[4])
                box2.append(b)

            for box, score in box2:
                boxes1.append(([LABELS, box], score))

            with open(prediction_file, 'w') as f:
                f.write(generateXML(fn, os.path.join(path, folder_name, fn), wI, hI, d, boxes1))
            print("Process Image" + " " + fn)-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # while True:
            #     cv2.imshow('res', raw_img)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
