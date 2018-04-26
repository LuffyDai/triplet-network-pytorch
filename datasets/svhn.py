from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
from torchvision.datasets.utils import download_url, check_integrity
from torchvision import transforms
from .base import Dataset as Base

class Dataset(Base):
    url = ""
    filename = ""
    file_md5 = ""
    num_classes = 10

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    name = 'svhn'

    def __init__(self, root, is_triplet=True,
                 train=True, transform=None, target_transform=None, download=True):
        self.root = os.path.expanduser(root)
        self.is_triplet = is_triplet
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self.target_transform = target_transform
        self.train=train
        
        if self.train:
            self.split = 'train'
        else:
            self.split = 'test'
        
        self.url = self.split_list[self.split][0]
        self.filename = self.split_list[self.split][1]
        self.file_md5 = self.split_list[self.split][2]
        
        if download:
            self.download()
        
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' + 
                               'You can use download=True to download it')
        
        import scipy.io as sio
        
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))
        
        if self.train:
            self.train_data = loaded_mat['X']
            self.train_labels = loaded_mat['y'].astype(np.int64).squeeze()
            np.place(self.train_labels, self.train_labels == 10, 0)
            self.train_data = np.transpose(self.train_data, (3, 2, 0, 1))
        else:
            self.test_data = loaded_mat['X']
            self.test_labels = loaded_mat['y'].astype(np.int64).squeeze()
            np.place(self.test_labels, self.test_labels == 10, 0)
            self.test_data = np.transpose(self.test_data, (3, 2, 0, 1))
        
        super(Dataset, self).__init__()

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def __getitem__(self, index):
        if self.is_triplet:
            if self.train:
                idx1, idx2, idx3 = self.triplets_train[index]
                img1, img2, img3 = self.train_data[idx1], self.train_data[idx2], self.train_data[idx3]
                target1, target2, target3 = self.train_labels[idx1], self.train_labels[idx2], self.train_labels[idx3]
            else:
                idx1, idx2, idx3 = self.triplets_test[index]
                img1, img2, img3 = self.test_data[idx1], self.test_data[idx2], self.test_data[idx3]
                target1, target2, target3 = self.test_labels[idx1], self.test_labels[idx2], self.test_labels[idx3]

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
                target = self.train_labels[index]
            else:
                img = self.test_data[index]
                target = self.test_labels[index]

            img = Image.fromarray(np.transpose(img, (1, 2, 0)))

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

            
        
        

