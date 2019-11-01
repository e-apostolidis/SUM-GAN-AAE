# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json

class VideoData(Dataset):
    def __init__(self, mode, split_index):
        self.mode = mode
        self.name = 'tvsum'
        self.datasets = ['../data/SumMe/eccv16_dataset_summe_google_pool5.h5',
                         '../data/TVSum/eccv16_dataset_tvsum_google_pool5.h5']
        self.splits_filename = ['../data/splits/' + self.name + '_splits.json']
        self.splits = []
        self.split_index = split_index # it represents the current split (varies from 0 to 4)
        temp = {}

        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]
        self.video_data = h5py.File(self.filename, 'r')

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for split in data:
                temp['train_keys'] = split['train_keys']
                temp['test_keys'] = split['test_keys']
                self.splits.append(temp.copy())

    def __len__(self):
        self.len = len(self.splits[0][self.mode+'_keys'])
        return self.len

    # In "train" mode it returns the features; in "test" mode it also returns the video_name
    def __getitem__(self, index):
        video_name = self.splits[self.split_index][self.mode + '_keys'][index]
        frame_features = torch.Tensor(np.array(self.video_data[video_name + '/features']))
        if self.mode == 'test':
            return frame_features, video_name
        else:
            return frame_features


def get_loader(mode, split_index):
    if mode.lower() == 'train':
        vd = VideoData(mode, split_index)
        return DataLoader(vd, batch_size=1)
    else:
        return VideoData(mode, split_index)


if __name__ == '__main__':
    pass
