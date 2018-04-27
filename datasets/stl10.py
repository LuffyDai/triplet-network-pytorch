from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import errno
import numpy as np
import sys
from torchvision.datasets.utils import download_url, check_integrity
from torchvision import transforms
from .base import Dataset as Base


class Dataset(Base):
    base_folder = 'stl10_binary'
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = '91f7769df0f17e558f3565bffb0c7dfb'
    class_names_file = 'class_names.txt'
    train_list = [
        ['train_X.bin', '918c2871b30a85fa023e0c44e0bee87f'],
        ['train_y.bin', '5a34089d4802c674881badbb80307741'],
        ['unlabeled_X.bin', '5242ba1fed5e4be9e1e742405eb56ca4']
    ]

    test_list = [
        ['test_X.bin', '7f263ba9f9e0b06b93213547f721ac82'],
        ['test_y.bin', '36f9794fa4beb8a2c72628de14fa638e']
    ]
    splits = ('train', 'train+unlabeled', 'unlabeled', 'test')
    num_classes = 10
    name = 'stl10'

    def __init__(self, root, is_triplet=True, train=True,
                 transform=None, target_transform=None, download=True):
        self.root = os.path.expanduser(root)
        self.is_triplet = is_triplet
        self.train = train
        if transform is None:
            self.transform  = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        else:
            self.transform = transform
        self.target_transform = target_transform
        if self.train:
            self.split = 'train'
        else:
            self.split = 'test'

        # download if need, same as cifar
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                'Dataset not found or corrupted. '
                'You can use download=True to download it'
            )
        if self.train:
            self.train_data, self.train_labels = self.__loadfile(
                self.train_list[0][0], self.train_list[1][0])
            self.n_train_triplets = self.train_data.shape[0]
        else:
            self.test_data, self.test_labels = self.__loadfile(
                self.test_list[0][0], self.test_list[1][0])
            self.n_test_triplets = self.test_data.shape[0]

        class_file = os.path.join(self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()
        super(Dataset, self).__init__()

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

    def __loadfile(self, data_file, labels_file=None):
        labels = None
        if labels_file:
            path_to_labels = os.path.join(
                self.root, self.base_folder, labels_file)
            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    def __getitem__(self, index):
        if self.is_triplet:
            if self.train:
                idx1, idx2, idx3 = self.triplets_train[index]
                img1, img2, img3 = self.train_data[idx1], self.train_data[idx2], self.train_data[idx3]
                target1, target2, target3 = int(self.train_labels[idx1]), int(self.train_labels[idx2]), int(self.train_labels[idx3])
            else:
                idx1, idx2, idx3 = self.triplets_test[index]
                img1, img2, img3 = self.test_data[idx1], self.test_data[idx2], self.test_data[idx3]
                target1, target2, target3 = int(self.test_labels[idx1]), int(self.test_labels[idx2]), int(self.test_labels[idx3])

            img1 = Image.fromarray(np.transpose(img1, (1, 2, 0)))
            img2 = Image.fromarray(np.transpose(img2, (1, 2, 0)))
            img3 = Image.fromarray(np.transpose(img3, (1, 2, 0)))

            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)

            if self.target_transform is not None:
                target1 = self.target_transform(target1)
                target2 = self.target_transform(target2)
                target3 = self.target_transform(target3)

            return (img1, target1), (img2, target2), (img3, target3)
        else:
            if self.train:
                img = self.train_data[index]
                target = int(self.train_labels[index])
            else:
                img = self.test_data[index]
                target = int(self.test_labels[index])

            img = Image.fromarray(np.transpose(img, (1, 2, 0)))

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target


