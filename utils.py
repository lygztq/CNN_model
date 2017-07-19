import cv2
import numpy as np
import os
import cPickle
import random

OUTPUT_SIZE = 2  # if True/False classification, than (F, T)

DATA_PATH = 'dataSet'

class dataSet(object):
    def __init__(self, images, labels, one_hot=False, dtype=np.float32):
        self._numOfData = len(labels)
        self.one_hot = one_hot
        self._data = images.astype(dtype)
        #
        # for i in range(self._data.shape[0]):
        #     for j in range(self._data.shape[1]):
        #         if self._data[i,j] == 255:
        #             self._data[i,j] = 1

        if not one_hot:
            self._label = np.array(labels)
        else:
            self._label = np.zeros((self._numOfData, OUTPUT_SIZE))
            for i in range(self._numOfData):
                self._label[i,labels[i]] = 1
        self._label = self._label.astype(dtype)

    def _get_batch_from_indices(self, indexs):
        dataSet = []
        labelSet = []
        for i in indexs:
            dataSet.append(np.array(self._data[i, :]))
            if self.one_hot:
                labelSet.append(self._label[i,:])
            else:
                labelSet.append(self._label[i])
        return np.array(dataSet), np.array(labelSet)

    def next_batch(self, batch_size):
        a, b = self._get_batch_from_indices(random.sample(xrange(self._numOfData), batch_size))
        batch = [a, b]
        return batch

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def numOfData(self):
        return self._numOfData
    

def read_data_set(path, one_hot=False):
    train_file_name = os.path.join(path,"batch_train.bin")
    test_file_name = os.path.join(path,"batch_test.bin")

    train_file = open(train_file_name, 'rb')
    test_file = open(test_file_name, 'rb')

    train_dict = cPickle.load(train_file)
    test_dict = cPickle.load(test_file)

    train_set = dataSet(train_dict['data'], train_dict['label'],one_hot=one_hot)
    test_set = dataSet(test_dict['data'], test_dict['label'], one_hot=one_hot)

    test_true_num = 0
    test_false_num = 0
    test_labels = test_dict['label']

    for i in test_labels:
        if i == 1:
            test_true_num += 1
        else:
            test_false_num += 1

    return train_set, test_set, test_true_num, test_false_num


