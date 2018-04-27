from __future__ import print_function
import torch.utils.data as data
import os
import numpy as np
from . import TripletDataset as Base


class TripletDataset(data.Dataset, Base):

    def __init__(self):
        self.train = self.base_dataset.train
        self.transform = self.base_dataset.transform

        if self.train:
            self.train_labels = self.base_dataset.train_labels
            self.train_data = self.base_dataset.train_data
            if isinstance(self.train_labels, list):
                self.labels_set = set(np.array(self.train_labels))
                self.label_to_indices = {label: np.where(np.array(self.train_labels) == label)[0] 
                                         for label in self.labels_set}
            elif isinstance(self.train_labels, np.ndarray):
                self.labels_set = set(self.train_labels)
                self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                         for label in self.labels_set}
            else:
                self.labels_set = set(self.train_labels.numpy())
                self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                         for label in self.labels_set}
        else:
            self.test_labels = self.base_datast.test_labels
            self.test_data = self.base_dataset.test_data
            # generate fixed triplets for testing
            if isinstance(self.test_labels, list):
                self.labels_set = set(np.array(self.test_labels))
                self.label_to_indices = {label: np.where(np.array(self.test_labels) == label)[0] 
                                         for label in self.labels_set}
            elif isinstance(self.test_labels, np.ndarray):
                self.labels_set = set(self.test_labels)
                self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                         for label in self.labels_set}
            else:
                self.labels_set = set(self.test_labels.numpy())
                self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                         for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i]]),
                         random_state.choice(self.label_to_indices[np.random.choice(
                             list(self.labels_set - set([self.test_labels[i]])))])]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[negative_index]
            label2 = self.train_labels[negative_index]
            img3 = self.train_data[positive_index]
            label3 = self.train_labels[positive_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][2]]
            img3 = self.test_data[self.test_triplets[index][1]]
            label1 = self.test_labels[self.test_triplets[index][0]]
            label2 = self.test_labels[self.test_triplets[index][2]]
            label3 = self.test_labels[self.test_triplets[index][1]]
        return (img1, label1), (img2, label2), (img3, label3)

    def __len__(self):
        return len(self.base_dataset)

    def get_base(self):
        return self.base_dataset

    @classmethod
    def parse(cls, text):
        if cls is not TripletDataset and text == cls.name:
            return cls(root=os.path.join('data', cls.name), train=True), \
                   cls(root=os.path.join('data', cls.name), train=False), \
                   cls(root=os.path.join('data', cls.name), train=True).get_base(), \
                   cls(root=os.path.join('data', cls.name), train=False).get_base()
