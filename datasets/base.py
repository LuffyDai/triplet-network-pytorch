from __future__ import print_function
import torch.utils.data as data
import os
import numpy as np
import csv

class TripletBase(data.Dataset):
    train_triplet_file = 'train_triplets.txt'
    test_triplet_file = 'test_triplets.txt'
    processed_folder = 'processed'

    def __init__(self):
        if self.train:
            self.make_triplet_list(self.n_train_triplets)
            triplets = []
            for line in open(os.path.join(self.root, self.processed_folder, self.train_triplet_file)):
                triplets.append((int(line.split()[0]), int(line.split()[1]), int(line.split()[2])))
            self.triplets_train = triplets
        else:
            self.make_triplet_list(self.n_test_triplets)
            triplets = []
            for line in open(os.path.join(self.root, self.processed_folder, self.test_triplet_file)):
                triplets.append((int(line.split()[0]), int(line.split()[1]), int(line.split()[2])))
            self.triplets_test = triplets

    def __len__(self):
        if self.train:
            return len(self.triplets_train)
        else:
            return len(self.triplets_test)

    def _check_triplets_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.train_triplet_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_triplet_file))

    def make_triplet_list(self, ntriplets):

        if self._check_triplets_exists():
            return
        print('Processing Triplet Generation ...')
        if self.train:
            np_labels = self.train_labels.numpy()
            filename = self.train_triplet_file
        else:
            np_labels = self.test_labels.numpy()
            filename = self.test_triplet_file
        triplets = []
        for class_idx in range(self.num_classes):
            a = np.random.choice(np.where(np_labels == class_idx)[0], int(ntriplets / self.num_classes), replace=True)
            b = np.random.choice(np.where(np_labels == class_idx)[0], int(ntriplets / self.num_classes), replace=True)
            while np.any((a-b) == 0):
                np.random.shuffle(b)
            c = np.random.choice(np.where(np_labels != class_idx)[0], int(ntriplets / self.num_classes), replace=True)

            for i in range(a.shape[0]):
                triplets.append([int(a[i]), int(c[i]), int(b[i])])

        with open(os.path.join(self.root, self.processed_folder, filename), "w") as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerows(triplets)
        print('Done!')


