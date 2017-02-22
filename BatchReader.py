from scipy import ndimage
from io import BytesIO
import threading
import numpy as np


def preprocess_img(image_data, resize, gray=False):
    hp = resize / image_data.shape[0]
    wp = resize / image_data.shape[1]
    image_data = ndimage.zoom(image_data, [hp, wp, 1], order=1)
    if gray:
        image_data = image_data[:, :, 0] * 0.2989 + \
                     image_data[:, :, 1] * 0.5870 + \
                     image_data[:, :, 2] * 0.114
    image_data = (image_data / 255.) - .5
    return image_data


class ThreadImgProcess(threading.Thread):
    def __init__(self, archive, image_size, gray, dataset, i, filename):
        threading.Thread.__init__(self)
        self.archive = archive
        self.image_size = image_size
        self.gray = gray
        self.dataset = dataset
        self.filename = filename
        self.i = i

    def run(self):
        while True:
            img_bytes = self.archive.read(self.filename)
            if len(img_bytes) > 0:  # wtf.. multi-threaded reads sometimes returned zero bytes.
                break
            print('cannot read file, retrying... ', self.filename)
        buff = BytesIO()
        buff.write(img_bytes)
        buff.seek(0)
        img = ndimage.imread(buff).astype(float)
        img = preprocess_img(img, self.image_size, self.gray)
        self.dataset[self.i] = img


class BatchReader:
    def __init__(self, archive, read_list, classifier):
        self.archive = archive
        self.read_list = read_list
        self.classifier = classifier
        self.count = len(read_list)

    def get_generator(self, batch_size=128, image_size=256, gray=False):
        if gray:
            dataset = np.ndarray(shape=(batch_size, image_size, image_size),
                                 dtype=np.float32)
        else:
            dataset = np.ndarray(shape=(batch_size, image_size, image_size, 3),
                                 dtype=np.float32)
        labset = np.ndarray(shape=(batch_size, self.classifier.num),
                            dtype=np.float32)

        threads = [None] * batch_size

        cnt = 0
        while True:
            print('reading from beginning: ', self.read_list[0])
            for file_i, v in enumerate(self.read_list):
                if v[-1] == '/':
                    print('skip:',  v)
                    continue
                threads[cnt] = ThreadImgProcess(self.archive, image_size, gray,
                                                dataset, cnt, v)
                threads[cnt].start()
                labset[cnt] = self.classifier.get(v)
                cnt += 1
                if cnt >= batch_size:
                    cnt = 0
                    for t in threads:
                        t.join()
                    threads = [None] * batch_size
                    # print('yield', file_i)
                    yield dataset, labset




