# -*- coding: utf-8 -*-
# @Author: User
# @Date:   2019-10-17 12:18:50
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-31 11:46:21
import os
import cv2
import time
from faced import FaceDetector

# filename = 'obama.jpg'
model = 'faced'
scale = 1
path = '../data/29--Students_Schoolkids/'

for fn in os.listdir(path):
    filename = fn
    raw_img = cv2.imread(os.path.join(path, filename))
    out_file = '../data'
    detector = FaceDetector()
    name = fn.split('.')
    name = name[0]
    out_file = os.path.join(out_file, name.replace('jpg', 'txt'))
    t0 = time.time()
    print('start')
    face_locations = detector.predict(raw_img, 0.5)
    t1 = time.time()
    print(f'took {round(t1-t0, 3)} to get {len(face_locations)} faces')

    for (x, y, w, h, _) in face_locations:
        x1 = x-int(w/2)
        x2 = x+int(w/2)
        y1 = y-int(h/2)
        y2 = y+int(h/2)
        # print(x1, y1, x2-x1, y2-y1)
        # cv2.rectangle(raw_img, (x - int(w / 2), y - int(h / 2)), (x + int(w / 2), y + int(h / 2)), (80, 18, 236), 2)
        with open(out_file + '.txt', 'a') as f:
            f.write("%s %g %d %d %d %d\n" % (str('face'), _ , x1, y1, x2-x1, y2-y1))
    # while True:
    #     cv2.imshow('IMG', raw_img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
