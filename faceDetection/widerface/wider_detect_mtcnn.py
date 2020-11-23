import cv2
from mtcnn.mtcnn import MTCNN
import os
import time

# filename = 'obama.jpg'
model = 'mtcnn'
scale = 1
path = '../data/29--Students_Schoolkids/'

for fn in os.listdir(path):
    detector = MTCNN()
    filename = fn
    raw_img = cv2.imread(os.path.join(path, filename))
    out_file = '../data'
    name = fn.split('.')
    name = name[0]
    out_file = os.path.join(out_file, name.replace('jpg', 'txt'))
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    print('start')
    face_locations = detector.detect_faces(img)
    t1 = time.time()
    print(f'took {round(t1-t0, 3)} to get {len(face_locations)} faces')
    for loc in face_locations:
        (x, y, w, h) = loc['box']
        confidence = loc['confidence']
        with open(out_file + '.txt', 'a') as f:
            f.write("%s %g %d %d %d %d\n" % (str('face'), confidence, x, y, w, h))
        # cv2.rectangle(raw_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # print(x, y, w, h)
        # print(confidence)
    # font = cv2.FONT_HERSHEY_DUPLEX
    # text = f'took {round(t1-t0, 3)} to get {len(face_locations)} faces'
    # cv2.putText(raw_img, text, (20, 20), font, 0.5, (255, 255, 255), 1)
    # cv2.imwrite(os.path.join('TestOutput', f'{model}_{scale}_{filename}'), raw_img)
    # while True:
    #     cv2.imshow('IMG', raw_img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
