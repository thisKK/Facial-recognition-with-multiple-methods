import glob
import os
import cv2
import time

from numba import prange
from faceDetection.face_detection_DSFD import face_detection


def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)


if __name__ == "__main__":

    path = '../widerface/wider/WIDER_val/images'
    out_file = '../widerface/wider/WIDER_prediction_DSFD'
    impaths = glob.glob(os.path.join(path, "*.jpg"))
    CONFIDENCE = 0.5
    LABELS = 'face'

    detector = face_detection.build_detector(
        "DSFDDetector",
        max_resolution=1080
    )
    for dir in os.listdir(path):
        os.mkdir(os.path.join(out_file, dir))
        print("Processing:", dir)

        for fn in os.listdir(os.path.join(path, dir)):
            raw_img = cv2.imread(os.path.join(path, dir, fn))
            t0 = time.time()
            dets = detector.detect(raw_img[:, :, ::-1])[:, :5]
            t1 = time.time()
            prediction_file = os.path.join(out_file, dir, fn.replace('jpg', 'txt'))
            name = fn.split('.')
            name = name[0]

            with open(prediction_file, 'w') as f:
                f.write("%s\n" % str(name))
            time.sleep(0.001)
            box2 = []
            for box in dets:
                if box[4].astype(float) < CONFIDENCE:
                    continue
                bbox = box[:4].astype(int)
                b = (bbox, box[4])
                box2.append(b)

            with open(prediction_file, 'a') as f:
                f.write("%d\n" % len(box2))
            for box, score in box2:
                with open(prediction_file, 'a') as f:
                    f.write("%d %d %d %d %g\n" % (box[0], box[1], box[2] - box[0], box[3] - box[1], score))
                time.sleep(0.001)
            print("Process Image" + " " + fn)
            # while True:
            #     cv2.imshow('IMG', raw_img)
            #     time.sleep(1000)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break