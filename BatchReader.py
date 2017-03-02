from scipy import ndimage
from io import BytesIO
import threading
import time
import gc
import weakref
import numpy as np
import multiprocessing

CPU_NUM = multiprocessing.cpu_count()


# 缩放zero mean
def preprocess_img(image_data, resize, gray=False):
    hp = resize / image_data.shape[0]
    wp = resize / image_data.shape[1]
    image_data = ndimage.zoom(image_data, [hp, wp, 1], order=1)
    if gray:
        image_data = image_data[:, :, 0] * 0.2989 + \
                     image_data[:, :, 1] * 0.5870 + \
                     image_data[:, :, 2] * 0.114
        image_data = image_data[:, :, None]
    image_data = (image_data / 255.) - .5
    return image_data


class BatchReader:
    def __init__(self, archive, read_list, classifier):
        self.archive = archive
        self.classifier = classifier
        self.dataset = None
        self.labels = np.ndarray(shape=(len(read_list), self.classifier.num),
                                 dtype=np.float32)
        self.read_list = []
        # 根据文件名生成labels，顺便跳过要忽略的项目
        cnt = 0
        for v in read_list:
            if v[-1] == '/' or v[-13:] == 'dog.10237.jpg':
                print('skip:',  v)
                continue
            self.labels[cnt] = classifier.get(v)
            self.read_list.append(v)
            cnt += 1
        self.labels = self.labels[:cnt]
        self.count = len(self.labels)
        self.loaded = 0

    def __del__(self):
        print('del ', self, self.count)

    def thread_reading(self, end, image_size=256, gray=False):
        if end <= self.loaded:
            return

        def thread_read(tid, ei):
            s = self
            for fi in range(self.loaded, ei):
                # 分布式
                if fi % CPU_NUM != tid:
                    continue
                v = s.read_list[fi]
                img_bytes = s.archive.read(v)
                buff = BytesIO()
                buff.write(img_bytes)
                buff.seek(0)
                img = ndimage.imread(buff).astype(float)
                img = preprocess_img(img, image_size, gray)
                s.dataset[fi] = img

        threads = [None] * CPU_NUM
        for i in range(CPU_NUM):
            threads[i] = threading.Thread(target=thread_read, args=[i, end])
            threads[i].start()
        for t in threads:
            t.join()
        self.loaded = end

    def get_generator(self, batch_size=128, image_size=256, gray=False):
        self.loaded = 0
        self.dataset = None
        self.dataset = np.ndarray(shape=(self.count, image_size, image_size, not gray and 3 or 1),
                                  dtype=np.float32)

        iter = 0
        cur = 0
        while True:
            step = min(batch_size, len(self.dataset) - cur)
            end = cur + step
            self.thread_reading(end, image_size, gray)
            rtn_dat = self.dataset[cur:end]
            rtn_lab = self.labels[cur:end]
            cur = end
            if step < batch_size:
                cur = 0
                step = batch_size - step
                end = step
                rtn_dat = np.vstack((rtn_dat, self.dataset[cur:end]))
                rtn_lab = np.vstack((rtn_lab, self.labels[cur:end]))
                cur = end
                iter += 1
                print('iter: ', iter)
            yield rtn_dat, rtn_lab




