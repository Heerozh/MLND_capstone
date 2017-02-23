import re
import numpy as np
from pathlib import Path
from imp import reload

import BatchReader
import ZipTarReader
BatchReader = reload(BatchReader)
ZipTarReader = reload(ZipTarReader)

PET_CLASS = {
    'chihuahua': 0,

    'japanese_spaniel': 1,
    'japanese_chin': 1,


}
PET_CLASS_NUM = 0  # todo
CLASSIFIER = re.compile('|'.join([re.escape(c) for c in PET_CLASS]))


class KaggleLabelClassifier:
    num = 1

    @staticmethod
    def get(filename):
        if re.search(r'dog', filename):
            return 1
        elif re.search(r'cat', filename):
            return 0
        else:
            return 2


class OxfordStanfordLabelClassifier:
    num = PET_CLASS_NUM

    @staticmethod
    def get(filename):
        return PET_CLASS[CLASSIFIER.search(filename).group()]


def get_kaggle_reader(validation_pct=.2):
    train_file = Path('./train.zip')
    test_file = Path('./test.zip')
    if not train_file.is_file() or not test_file.is_file():
        print('👇👇👇👇👇👇👇👇👇👇👇👇👇')
        print('Please put train.zip and test.zip from '
              'https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition to project root path.')
        raise FileNotFoundError()

    train_arc = ZipTarReader.auto(train_file.name)
    test_arc = ZipTarReader.auto(test_file.name)

    file_list = train_arc.shuffled_namelist()
    file_list = np.array(file_list)
    mid = int(len(file_list) * validation_pct)
    train_file_list = file_list[mid:]
    vali_file_list = file_list[:mid]

    return BatchReader.BatchReader(train_arc, train_file_list, KaggleLabelClassifier),\
        BatchReader.BatchReader(train_arc, vali_file_list, KaggleLabelClassifier),\
        BatchReader.BatchReader(test_arc, test_arc.shuffled_namelist(), KaggleLabelClassifier)






