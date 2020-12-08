import cv2
import os
import time
import numpy as np

model = 'Yolov3'

path = '../widerface/wider/WIDER_val/images'
out_file = '../widerface/wider/WIDER_prediction_YOLO'

YoloConfig = '../face_detection_yolov3/Yolo/yolo_models/yolov3-face.cfg'
YoloWeight = '../face_detection_yolov3/Yolo/yolo_weights/yolov3-face.weights'

count = 0
CONFIDENCE = 0.5
THRESH = 0.3
IMG_WIDTH, IMG_HEIGHT = 416, 416

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
            h, w, _ = raw_img.shape
            blob = cv2.dnn.blobFromImage(raw_img, 1/255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)

            net.setInput(blob)
            layers_names = net.getLayerNames()
            outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])

            prediction_file = os.path.join(out_file, folder_name, fn.replace('jpg', 'txt'))
            name = fn.split('.')
            name = name[0]
            with open(prediction_file, 'w') as f:
                f.write("%s\n" % str(name))

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

                        bbox = np.array([x, y, int(bwidth), int(bheight)])
                        b = (bbox, confidence)
                        box2.append(b)

                        # cropped = raw_img[y:y+bheight, x:x+bwidth]
                        # face = cv2.resize(cropped, (112, 112))
                        # cv2.imwrite("./cropped_face/face_" + str(count) + ".jpg", newimg)

            # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESH)

            t1 = time.time()
            # print(f"took {round(t1-t0, 3)} to get {len(idxs.flatten())} faces")

            if boxes is None or confidences is None or idxs is None or class_ids is None:
                raise '[ERROR] Required variables are set to None before drawing boxes on images.'

            with open(prediction_file, 'a') as f:
                f.write("%d\n" % len(idxs))

            # for box, score in box2:
            #     cv2.rectangle(raw_img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (80, 18, 236), 2)
            #     with open(prediction_file, 'a') as f:
            #         f.write("%d %d %d %d %g\n" % (box[0], box[1], box[2], box[3], score))
            #     time.sleep(0.001)
            if len(idxs) > 0:
                for i in idxs.flatten():
                    x, y = boxes[i][0], boxes[i][1]
                    w, h = boxes[i][2], boxes[i][3]
                    cv2.rectangle(raw_img, (x,y), (x+w,y+h), (80,18,236), 2)
                    with open(prediction_file, 'a') as f:
                        f.write("%d %d %d %d %g\n" % (x, y, w, h, confidences[i]))
            print("Process Image" + " " + name)

            # while True:
            #     cv2.imshow('IMG', raw_img)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
