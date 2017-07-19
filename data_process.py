import cv2
import numpy as np
import os
import cnn_train
import cnn
import cPickle
import random

# Some constants and parameters
IMG_SIZE = cnn.get_default_hparams().input_img_size
source_path = os.path.join(cnn_train.project_path, 'ResultPNG')
target_path = os.path.join(cnn_train.project_path, 'ResultPNGGray')
binDataSetPath = os.path.join(cnn_train.project_path, 'dataSet')


def color2gray(img):
    """Change colorful pictures to gray degree picture."""
    b = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    r = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)

    b[:, :] = img[:, :, 0]
    g[:, :] = img[:, :, 1]
    r[:, :] = img[:, :, 2]

    ret = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ret[i, j] = b[i, j]/3+g[i, j]/3+b[i, j]/3  # to avoid overflow

    return ret


def save_img(img, path):
    """Saving img."""
    cv2.imwrite(path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


def black_and_white(img, boundary=255):
    """Change picture to a black-and-white picture."""
    ans = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < boundary:
                ans[i, j] = 0
            else:
                ans[i, j] = 255
    return ans


def change_size(img, target_size=IMG_SIZE):
    """Change the size of pictures."""
    if target_size < 128:
        img = black_and_white(img, 255)
        res = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        res = black_and_white(res, 200)
    else:
        res = black_and_white(img, 200)
    return res


def get_np_array(img):
    """Change picture into numpy array."""
    ans = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ans.append(img[i, j])
    ans = np.array(ans)
    return ans


def get_name_list(path):
    """Get the name list of pictures in a directory."""
    ans = None
    for _, _, filename in os.walk(path):
        ans = filename
    return ans


def random_sample():
    """Randomly sampling pictures."""
    probability = np.random.uniform(0, 1, size=[1])
    probability = probability[0] - 0.8
    if probability <= 0:
        return False
    else:
        return True


def two_value(img):
    """Change to 0-1 data."""
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 255:
                img[i, j] = 1
    return img


def divide_data(path):
    """Divide data into train set and test set."""
    true_file = open(os.path.join(path, "good.txt"))
    false_file = open(os.path.join(path, "bad.txt"))

    true_file_content = true_file.read().split()
    false_file_content = false_file.read().split()

    numOfTrue = len(true_file_content)
    numOfFalse = len(false_file_content)

    random.shuffle(true_file_content)
    random.shuffle(false_file_content)

    true_test_set = true_file_content[0:int(numOfTrue/5)]
    true_train_set = true_file_content[int(numOfTrue/5):]
    false_test_set = false_file_content[0:int(numOfTrue/5)]
    false_train_set = false_file_content[int(numOfTrue/5):numOfTrue]

    train_img_set = []
    train_label_set = []
    test_img_set = []
    test_label_set = []

    print "num of true case in train set: ", len(true_train_set)
    print "num of true case in test set: ", len(true_test_set)
    print "num of false case in train set: ", len(false_train_set)
    print "num of false case in test set: ", len(false_test_set)

    for i in range(len(true_train_set)):
        filename = true_train_set[i] + ".png"
        filepath = os.path.join(path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = change_size(img)
        img = two_value(img)
        img_array = get_np_array(img)
        train_img_set.append(img_array)
        train_label_set.append(1)

    for i in range(len(false_train_set)):
        filename = false_train_set[i] + ".png"
        filepath = os.path.join(path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = change_size(img)
        img = two_value(img)
        img_array = get_np_array(img)
        train_img_set.append(img_array)
        train_label_set.append(0)

    for i in range(len(true_test_set)):
        filename = true_test_set[i] + ".png"
        filepath = os.path.join(path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = change_size(img)
        img = two_value(img)
        img_array = get_np_array(img)
        test_img_set.append(img_array)
        test_label_set.append(1)

    for i in range(len(false_test_set)):
        filename = false_test_set[i] + ".png"
        filepath = os.path.join(path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = change_size(img)
        img = two_value(img)
        img_array = get_np_array(img)
        test_img_set.append(img_array)
        test_label_set.append(0)

    return train_img_set, train_label_set, test_img_set, test_label_set


def read_data_set(path):
    img_set = []
    true_name_set = set()
    false_name_set = set()

    file_name_set = get_name_list(path)

    for filename in file_name_set:
        if filename[-3:] == "png":
            filepath = os.path.join(path, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img_name_pair = [img, filename]
            img_set.append(img_name_pair)
        elif filename == "good.txt":
            filepath = os.path.join(path, filename)
            f = open(filepath, 'r')
            file_content = f.read()
            file_content = file_content.split()
            for i in file_content:
                true_name_set.add(i)
        else:
            filepath = os.path.join(path, filename)
            f = open(filepath, 'r')
            file_content = f.read()
            file_content = file_content.split()
            for i in file_content:
                false_name_set.add(i)

    return img_set, true_name_set, false_name_set


def change_size_process(source_path, target_path):
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # test_label_set = []
    # test_img_set = []
    # train_label_set = []
    # train_img_set = []

    dir_name_set = []
    for _, dirs, _ in os.walk(source_path):
        for dir in dirs:
            dir_name_set.append(dir)

    cnt = 0
    for dir_name in dir_name_set:
        print cnt, dir_name
        dir_path = os.path.join(source_path, dir_name)
        target_dir_path = os.path.join(target_path, dir_name)
        if not os.path.exists(target_dir_path):
            os.mkdir(target_dir_path)
        #
        # img_set,true_names, false_names = read_data_set(dir_path)
        # for img in img_set:
        #     img[0] = change_size(img[0])
        #     img_name = img[1].split('.')
        #     img_name = img_name[0]
        #     img[0] = two_value(img[0])
        #     img_array = get_np_array(img[0])

            # if random_sample():
            #     if img_name in true_names:
            #         test_label_set.append(1)
            #         test_img_set.append(img_array)
            #     elif img_name in false_names:
            #         test_label_set.append(0)
            #         test_img_set.append(img_array)
            # else:
            #     if img_name in true_names:
            #         train_label_set.append(1)
            #         train_img_set.append(img_array)
            #     elif img_name in false_names:
            #         train_label_set.append(0)
            #         train_img_set.append(img_array)
            # target_img_path = os.path.join(target_dir_path,img[1])
            # save_img(img[0], target_img_path)
        train_img_set, train_label_set, test_img_set, test_label_set = divide_data(dir_path)
        cnt += 1

    train_img_set = np.array(train_img_set)
    test_img_set = np.array(test_img_set)

    train_dict = {'label': train_label_set, 'data': train_img_set}
    test_dict = {'label': test_label_set, 'data': test_img_set}
    train_bin_file = open(os.path.join(binDataSetPath, "batch_train.bin"), 'wb')
    test_bin_file = open(os.path.join(binDataSetPath, "batch_test.bin"), 'wb')
    cPickle.dump(test_dict, test_bin_file)
    cPickle.dump(train_dict, train_bin_file)


if __name__ == "__main__":
    change_size_process(source_path, target_path)
