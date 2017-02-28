import re
import tarfile
import zipfile
import random


class ArcBaseReader:
    def namelist(self):
        pass

    def read(self, fullname):
        pass

    def shuffled_namelist(self):
        namelist = self.namelist()
        random.shuffle(namelist)
        return namelist


class ZipReader(ArcBaseReader):
    def __init__(self, filename):
        self.filename = filename
        self.zip = zipfile.ZipFile(self.filename, 'r')

    def namelist(self):
        file_list = self.zip.namelist()
        digits = re.compile(r'(\d+)')

        def tokenize(filename):
            return tuple(int(token) if match else token
                         for token, match in
                         ((fragment, digits.search(fragment))
                          for fragment in digits.split(filename)))
        file_list = sorted(file_list, key=tokenize)
        return file_list

    def read(self, fullname):
        return self.zip.read(fullname)


class TarReader(ArcBaseReader):
    def __init__(self, filename):
        self.filename = filename
        self.tar = tarfile.open(self.filename, 'r')

    def namelist(self):
        file_list = []
        for member in self.tar.getmembers():
            if member.isfile():
                file_list.append(member.name)
        return file_list

    def read(self, fullname):
        f = self.tar.extractfile(fullname)
        return f.read()


def auto(filename):
    if re.search(r'zip$', filename):
        return ZipReader(filename)
    else:
        return TarReader(filename)


