from __future__ import print_function
import os
import torch


def _register(impl):
    _register.impls.append(impl)


_register.impls = []


class Dataset(object):

    class __metaclass__(type):

        def __init__(cls, name, bases, fields):
            type.__init__(cls, name, bases, fields)
            _register(cls)

    @classmethod
    def from_name(cls, name):
        inst = cls.parse(name)
        assert inst is not None, 'Unknown dataset {}'.format(name)
        return inst

    @classmethod
    def parse(cls, text):
        if cls is Dataset:
            for impl in _register.impls:
                if impl is not Dataset:
                    inst = impl.parse(text)
                    if inst is not None:
                        return inst


def get_Dataset(name, batch_size, **kwargs):
    train_triplet_set, test_triplet_set, train_set, test_set = Dataset.from_name(name)
    train_triplet_loader = torch.utils.data.DataLoader(
        train_triplet_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_triplet_loader = torch.utils.data.DataLoader(
        test_triplet_set, batch_size=batch_size, shuffle=False, **kwargs)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, **kwargs)
    return train_triplet_loader, test_triplet_loader, \
           train_loader, test_loader


