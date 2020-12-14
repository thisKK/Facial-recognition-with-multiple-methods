import cv2
import numpy as np
import os
import time
import glob
from faceDetection import RetinaFace

model = 'resnet50'
name = 'retinaFace'
CONFIDENCE = 0.1

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

if __name__ == "__main__":
    impaths = os.path.join('../../testImage/')
    impaths = glob.glob(os.path.join(impaths, "*.jpg"))
    detector = RetinaFace(gpu_id=0)

    for impath in impaths:
        count = 0
        if impath.endswith("out.jpg"): continue
        im = cv2.imread(impath)
        print("Processing:", impath)
        t = time.time()
        faces = detector(im)
        print(f"Detection time: {time.time() - t:.3f}")
        for box, landmarks, score in faces:
            if score.astype(float) < CONFIDENCE:
                continue
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
            count =count+1
        print("face found: ", count)
        imname = os.path.basename(impath).split(".")[0]
        output_path = os.path.join(
            os.path.dirname(impath),
            f"{imname}_out.jpg"
        )

        cv2.imwrite(output_path, im)

