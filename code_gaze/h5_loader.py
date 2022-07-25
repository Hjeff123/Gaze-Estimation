# coding: utf-8

import os
import numpy as np
import h5py

import torch
import torch.utils.data


class MPIIGazeDataset(torch.utils.data.Dataset):
    def __init__(self, subject_id, dataset_dir):
        # print(subject_id)
        ext = ('{}.h5')
        path = os.path.join(dataset_dir, ext.format(subject_id))
        print('Loading data in ', path, ',Pls waiting')
        with h5py.File(path, 'r') as fin:
            self.images = fin['data'][:]
            self.gazes = fin['label'][:, 0:2]
            self.poses = fin['label'][:, 2:4]
            # self.landmarks = fin['label'][:, 4:]
            # print(self.images.shape)
        self.length = len(self.images)

        #self.images = torch.unsqueeze(torch.from_numpy(self.images), 1) # unsqueeze 增加一个维度. input resolution: 36x60, channel: 1
        # self.poses = torch.from_numpy(self.poses)
        # self.gazes = torch.from_numpy(self.gazes)
        fin.close()

    def __getitem__(self, index):
        # return self.images[index], self.poses[index], self.gazes[index]
        return self.images[index], self.gazes[index], self.poses[index]

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__


def get_loader(dataset_dir, train_subject_id, test_subject_id, batch_size, num_workers, use_gpu):
    assert os.path.exists(dataset_dir)
    # assert test_subject_id in range(15)
    subject_ids=[]
    for i in range(10):
        subject_ids =  np.append([('p{:02}'.format(index)+'_'+str(i)) for index in range(15)], subject_ids)
    # print(subject_ids)
    # print(train_subject_id[0])
    train_subject_id_sub = [1]
    for k in range(len(train_subject_id)):
        for j in range (train_subject_id[k],train_subject_id[k]+1):
            if k==0:
                for i in range(1,10):
                    train_subject_id_sub=np.append(train_subject_id_sub, j+i*15)
            else:
                for i in range(0,10):
                    train_subject_id_sub=np.append(train_subject_id_sub, j+i*15)
    # print(test_subject_id)
    # print(train_subject_id_sub)
    # test_subject_id = [subject_ids[id] for id in test_subject_id]
    train_subject_id = [subject_ids[id] for id in train_subject_id_sub]
    print('Train data names:', train_subject_id)
    train_dataset = torch.utils.data.ConcatDataset([
        MPIIGazeDataset(subject_id, dataset_dir) for subject_id in train_subject_id])
    print('Samples of training data:', len(train_dataset))

    test_subject_id = 'p' + str('%02d' % test_subject_id)
    test_subject_id_sub=[]
    for i in range(10):
        test_subject_id_sub = np.append((test_subject_id + '_' + str(i)), test_subject_id_sub)
    print('Test data names', test_subject_id_sub)
    test_dataset = torch.utils.data.ConcatDataset([
        MPIIGazeDataset(subject_id, dataset_dir) for subject_id in test_subject_id_sub])
    print('Samples of testing data:', len(test_dataset))

    # assert len(train_dataset) == 42000
    # assert len(test_dataset) == 3000
    # print(len(train_dataset))
    # print(len(test_dataset))

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