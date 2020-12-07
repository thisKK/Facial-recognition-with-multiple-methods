import cv2
import numpy as np
import os
import time
import pickle
from face_detection import RetinaFace
from faceDetection.face_detection_reatinaface.detector import RetinaFace

path = '../widerface/wider/WIDER_val/images'
out_file = '../widerface/wider/WIDER_prediction'

model = 'resnet50'
count = 0
CONFIDENCE = 0.5
if __name__ == "__main__":
    detector = RetinaFace(gpu_id=0)
    for dir in os.listdir(path):
        folder_name = dir
        os.mkdir(os.path.join(out_file, folder_name))
        print(folder_name)
        for fn in os.listdir(os.path.join(path, folder_name)):
            raw_img = cv2.imread(os.path.join(path, folder_name, fn))
            t0 = time.time()
            faces = detector(raw_img)
            t1 = time.time()

            prediction_file = os.path.join(out_file, folder_name, fn.replace('jpg', 'txt'))
            box2 = []
            name = fn.split('.')
            name = name[0]
            with open(prediction_file, 'w') as f:
                f.write("%s\n" % str(name))
            time.sleep(0.001)

            for box, landmarks, score in faces:
                box = box.astype(np.int)
                if score.astype(np.float) > CONFIDENCE:
                    # cv2.rectangle(raw_img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)
                    b = (box, score)
                    box2.append(b)
            with open(prediction_file, 'a') as f:
                f.write("%d\n" % len(box2))
            for box, score in box2:
                # print(box)
                # print("bbox", box[0],box[1],box[2],box[3])
                # print(score)
                with open(prediction_file, 'a') as f:
                    f.write("%d %d %d %d %g\n" % (box[0], box[1], box[2] - box[0], box[3] - box[1], score))
                time.sleep(0.001)

            # while True:
            #     cv2.imshow('IMG', raw_img)
            #     # time.sleep(1000)
            #     # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     #     break

