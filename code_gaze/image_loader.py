# coding: utf-8

import os
import numpy as np
import cv2

import torch
import torch.utils.data

from PIL import Image


class MPIIGazeDataset(torch.utils.data.Dataset):
    def __init__(self, subject_id, dataset_dir):
        # print(subject_id)
        ext = ('{}images/')
        path = os.path.join(dataset_dir, ext.format(subject_id))
        print('Loading image data in ', subject_id, ',Pls waiting')
        self.images = np.array([x.path for x in os.scandir(path) if x.name.endswith('.jpg')]) # jpg
        ext = ('{}gazes/')
        path = os.path.join(dataset_dir, ext.format(subject_id))
        # print('Loading gaze data in ', subject_id, ',Pls waiting')
        self.gazes = np.array([x.path for x in os.scandir(path) if x.name.endswith('.txt')])
        ext = ('{}poses/')
        path = os.path.join(dataset_dir, ext.format(subject_id))
        # print('Loading pose data in ', subject_id, ',Pls waiting')
        self.poses = np.array([x.path for x in os.scandir(path) if x.name.endswith('.txt')])
        # print(self.poses)
        self.length = len(self.images)
        # print(self.length)

        #self.images = torch.unsqueeze(torch.from_numpy(self.images), 1) # unsqueeze 增加一个维度. input resolution: 36x60, channel: 1
        # self.poses = torch.from_numpy(self.poses)
        # self.gazes = torch.from_numpy(self.gazes)
        # fin.close()

    def __getitem__(self, index):
        # return self.images[index], self.poses[index], self.gazes[index]
        images = np.array(Image.open(self.images[index])) / 255
        images = cv2.resize(images, (448, 448))
        images = images.transpose(2, 0, 1) # 把颜色通道提前
        # print(images.shape)
        return images, np.loadtxt(self.gazes[index]), np.loadtxt(self.poses[index])

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__


def get_loader(dataset_dir, train_subject_id, test_subject_id, batch_size, num_workers, use_gpu):
    # assert os.path.exists(dataset_dir)
    # assert test_subject_id in range(15)
    subject_ids = ['p{:02}/'.format(index) for index in range(35)] # MPIIGaze 15, EyeDiap 14, UTMultiview 50
    # print(subject_ids)
    test_subject_id = subject_ids[test_subject_id]

    train_dataset = torch.utils.data.ConcatDataset([
        MPIIGazeDataset(subject_id, dataset_dir) for subject_id in subject_ids
        if subject_id != test_subject_id])

    test_dataset = MPIIGazeDataset(test_subject_id, dataset_dir)

    #    assert len(train_dataset) == 42000
    #   assert len(test_dataset) == 3000

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader