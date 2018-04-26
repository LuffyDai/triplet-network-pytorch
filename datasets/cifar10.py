from __future__ import print_function
from PIL import Image
import os
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from .base import TripletBase

class T_CIFAR10(TripletBase):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    num_classes = 10
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    n_train_triplets = 50000
    n_test_triplets = 10000

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               'You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

                self.train_data = np.concatenate(self.train_data)
                self.train_data = self.train_data.reshape((self.n_train_triplets, 3, 32, 32))
                self.train_data = self.train_data.transpose((0, 2, 3, 1))
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((self.n_test_triplets, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3 ,1))
        super(T_CIFAR10, self).__init__()

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), 'r:gz')
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __getitem__(self, index):
        if self.train:
            idx1, idx2, idx3 = self.triplets_train[index]
            img1, img2, img3 = self.train_data[idx1], self.train_data[idx2], self.train_data[idx3]
            target1, target2, target3 = self.train_labels[idx1], self.train_labels[idx2], self.train_labels[idx3]
        else:
            idx1, idx2, idx3 = self.triplets_test[index]
            img1, img2, img3 = self.test_data[idx1], self.test_data[idx2], self.test_data[idx3]
            target1, target2, target3 = self.test_labels[idx1], self.test_labels[idx2], self.test_labels[idx3]

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img3 = Image.fromarray(img3)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        if self.target_transform is not None:
            target1 = self.target_transform(target1)
            target2 = self.target_transform(target2)
            target3 = self.target_transform(target3)

            return (img1, target1), (img2, target2), (img3, target3)




