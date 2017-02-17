import zipfile
import random
import re
from scipy import ndimage
from pathlib import Path
from io import BytesIO
import threading

import numpy as np


TRAIN_IMAGES = './train.zip'
TEST_IMAGES = './test.zip'


def preprocess_img(image_data, resize, gray=False):
    hp = resize / image_data.shape[0]
    wp = resize / image_data.shape[1]
    image_data = ndimage.zoom(image_data, [hp, wp, 1], order=1)
    if gray:
        image_data = image_data[:,:,0] * 0.2989 + image_data[:,:,1] * 0.5870 + image_data[:,:,2] * 0.114
    image_data = (image_data / 255.) - .5
    return image_data


class ThreadRader (threading.Thread):
    def __init__(self, zfile, image_size, gray, dataset, labset, i, filename):
        threading.Thread.__init__(self)
        self.zfile = zfile
        self.image_size = image_size
        self.gray = gray
        self.dataset = dataset
        self.labset = labset
        self.filename = filename
        self.i = i

    def run(self):
        img_bytestr = self.zfile.read(self.filename)
        buff = BytesIO()
        buff.write(img_bytestr)
        buff.seek(0)
        img = ndimage.imread(buff).astype(float)
        img = preprocess_img(img, self.image_size, self.gray)
        self.dataset[self.i] = img
        self.labset[self.i] = re.search(r'dog', self.filename) and 1 or 0


def reader(zfile, flist, batch_size, image_size, gray):
    if gray:
        dataset = np.ndarray(shape=(batch_size, image_size, image_size),
                             dtype=np.float32)
    else:
        dataset = np.ndarray(shape=(batch_size, image_size, image_size, 3),
                             dtype=np.float32)
    labset = np.ndarray(shape=(batch_size),
                        dtype=np.float32)

    threads = [None] * batch_size

    cnt = 0
    for v in flist:
        threads[cnt] = ThreadRader(zfile, image_size, gray, dataset, labset, cnt, v)
        threads[cnt].start()
        cnt += 1
        if cnt >= batch_size:
            cnt = 0
            for t in threads:
                t.join()
            threads = [None] * batch_size
            yield dataset, labset
    dataset = dataset[:cnt]
    labset = labset[:cnt]
    yield dataset, labset


def shuffle_reader(filename, batch_size, image_size, gray=False, file_list=None):
    with zipfile.ZipFile(filename, 'r') as zfile:
        flist = file_list is None and zfile.namelist() or file_list
        random.shuffle(flist)
        rtn = reader(zfile, flist, batch_size, image_size, gray)
        for item in rtn:
            yield item


def read_data_sets(train_dir, batch_size=128, image_size=512, vali_pct=.2, gray=False):
    train_file = Path(TRAIN_IMAGES)
    test_file = Path(TEST_IMAGES)
    if not train_file.is_file() or not test_file.is_file():
        print('👇👇👇👇👇👇👇👇👇👇👇👇👇')
        print('Please put train.zip and test.zip from '
              'https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition to project root path.')
        raise FileNotFoundError()

    with zipfile.ZipFile(TRAIN_IMAGES, 'r') as zfile:
        flist = zfile.namelist()
        random.shuffle(flist)
        flist = np.array(flist)
        mid = int(len(flist) * vali_pct)
        train_filelist = flist[:mid]
        vali_filelist = flist[mid:]

    return shuffle_reader(TRAIN_IMAGES, batch_size, image_size, gray, train_filelist), \
        shuffle_reader(TRAIN_IMAGES, batch_size, image_size, gray, vali_filelist), \
        shuffle_reader(TEST_IMAGES, batch_size, image_size, gray)
