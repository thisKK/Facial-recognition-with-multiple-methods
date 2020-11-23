import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import os
import time
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

model = 'mtcnn'
scale = 1
path = '../data/29--Students_Schoolkids/'
CONFIDENCE = 0.5
LABELS = 'face'
output_filepath = '../data'

for fn in os.listdir(path):
    detector = MTCNN()
    filename = fn
    raw_img = cv2.imread(os.path.join(path, filename))
    out_file = '../data'
    wI, hI, d = raw_img.shape
    name = fn.split('.')
    name = name[0]
    out_file = os.path.join(out_file, name.replace('jpg', 'xml'))
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    print('start')
    face_locations = detector.detect_faces(img)
    t1 = time.time()
    print(f'took {round(t1-t0, 3)} to get {len(face_locations)} faces')
    boxes1 = []

    for loc in face_locations:
        (x, y, w, h) = loc['box']
        box = loc['box']
        box = np.asarray(box, dtype=np.float32)
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        confidence = loc['confidence']
        if float(confidence) < CONFIDENCE:
            continue
        boxes1.append(([LABELS, box], confidence))
    with open(out_file + '.xml', 'w') as f:
        f.write(generateXML(filename, output_filepath, hI, wI, d, boxes1))

    # while True:
    #     cv2.imshow('IMG', raw_img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
