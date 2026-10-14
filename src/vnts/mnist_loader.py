"""
Fast zero-dependency MNIST loader reading directly from gzip files into PyTorch tensors.
"""

import gzip
import numpy as np
import torch
import os

def load_mnist_data(data_dir="/root/vnts_clone/data/mnist", n_train=10000, n_test=2000):
    train_img_path = os.path.join(data_dir, "train-images.gz")
    train_lbl_path = os.path.join(data_dir, "train-labels.gz")
    test_img_path = os.path.join(data_dir, "test-images.gz")
    test_lbl_path = os.path.join(data_dir, "test-labels.gz")

    with gzip.open(train_img_path, 'rb') as f:
        f.read(16) # Skip magic and header
        train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28 * 28)
    
    with gzip.open(train_lbl_path, 'rb') as f:
        f.read(8)
        train_labels = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(test_img_path, 'rb') as f:
        f.read(16)
        test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28 * 28)

    with gzip.open(test_lbl_path, 'rb') as f:
        f.read(8)
        test_labels = np.frombuffer(f.read(), dtype=np.uint8)

    # Normalize to [0, 1] float32
    train_X = torch.from_numpy(train_images[:n_train].astype(np.float32) / 255.0)
    train_y = torch.from_numpy(train_labels[:n_train].astype(np.int64))
    test_X = torch.from_numpy(test_images[:n_test].astype(np.float32) / 255.0)
    test_y = torch.from_numpy(test_labels[:n_test].astype(np.int64))

    return (train_X, train_y), (test_X, test_y)
