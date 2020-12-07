import cv2
import os
import time
import tensorflow as tf
from lib.core.api.face_detector import FaceDetector
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from imutils import paths

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
detector = FaceDetector(['./model/detector.pb'])

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

def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # if s == "pts":
            #     continue
            newDir=os.path.join(dir, s)
            GetFileList(newDir, fileList)
    return fileList


def facedetect():
    count = 0
    data_dir = './data/29--Students_Schoolkids'
    pics = []
    GetFileList(data_dir, pics)
    CONFIDENCE = 0.1
    pics = [x for x in pics if 'jpg' in x or 'png' in x]
    LABELS = 'face'
    #pics.sort()
    output_filepath = './data'

    for pic in pics:
        filename = pic
        v = filename.split('/')
        v = v[3]
        v = str(v)
        v = v.split('.')
        v = v[0]
        out_file_path = os.path.join('./data/', v)
        img = cv2.imread(pic)
        wI, hI, d = img.shape
        img_show = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out_file = './data/'
        name = filename.split('/')
        name = name[3]
        out_file = os.path.join(out_file, name.replace('jpg', 'txt'))
        t0 = time.time()
        print('start')
        boxes = detector(img, 0.5)
        t1 = time.time()
        print(f'took {round(t1 - t0, 3)} to get {len(boxes)} faces')
        print(boxes.shape[0])
        boxes1 = []
        if boxes.shape[0]==0:
            print(pic)
        for box_index in range(boxes.shape[0]):
            bbox = boxes[box_index]
            box = bbox
            box = np.delete(box, 4)
            box = box.astype(np.int)
            score = float(bbox[4])
            x = int(bbox[0])
            y = int(bbox[1])
            w = int(bbox[2])
            h = int(bbox[3])
            if score < CONFIDENCE:
                continue
            boxes1.append(([LABELS, box], score))
            with open(out_file_path + '.xml', 'w') as f:
                f.write(generateXML(v, output_filepath, hI, wI, d, boxes1))
            # cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])),
            #               (int(bbox[2]), int(bbox[3])), (255, 0, 0), 4)
        # while True:
        #     cv2.imshow('res', img_show)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
    print(count)

if __name__=='__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    facedetect()
