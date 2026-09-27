"""
Dataset registry — CIFAR-10 only, by design. Same interface
(get_dataset / get_raw / get_raw_train / build_transforms /
num_classes_for / DATASET_SPECS) as the multi-dataset version, so
run_experiment.py, registry_poison.py etc. need zero changes. Add
another dataset later by adding one more entry here, same pattern.

CIFAR-10 stores its data as 5 pickled batch files (~170MB total) rather
than tens of thousands of individual images — this is exactly why it
doesn't hit the slow-many-small-files problem GTSRB/Tiny-ImageNet did,
and why it downloads and trains fast on fresh local Colab disk every
session with no persistent storage needed at all.
"""
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

DATASET_SPECS = {
    "cifar10": {"num_classes": 10, "in_channels": 3, "image_size": 32},
}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(name):
    assert name == "cifar10"
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, eval_tf


class IndexedWrapper(Dataset):
    """Makes the base dataset return (image, label, index) — the
    convention used everywhere else in the project."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return image, label, index


def get_raw(name, data_root, train, transform=None, download=True):
    assert name == "cifar10"
    return torchvision.datasets.CIFAR10(root=data_root, train=train, download=download, transform=transform)


def get_raw_train(name, data_root, download=True):
    return get_raw(name, data_root, train=True, transform=None, download=download)


def get_dataset(name, data_root, train, transform, download=True):
    return IndexedWrapper(get_raw(name, data_root, train, transform, download))


def num_classes_for(name, data_root=None):
    return DATASET_SPECS[name]["num_classes"]
