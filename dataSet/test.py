import cv2
import numpy as np
import os
import cPickle

index = 5

file = open("batch_train.bin",'rb')

dict = cPickle.load(file)

imgs = dict['data']
labels = dict['label']

test_img = imgs[index, :]

show_img = np.zeros((48,48))

for i in range(48):
    for j in range(48):
        show_img[i,j] = test_img[i*48+j]
cv2.imshow("test",show_img)
print labels
cv2.waitKey(0)
cv2.destroyAllWindows()