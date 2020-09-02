import argparse
import os
import time
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils import data
import pickle
import numpy as np

def load_imgs(file_path):
    sample_rate = 10
    all_imgs = []
    with open(file_path, 'r') as fr:
        for line in fr:
            splits = line.split(',')
            start_frame = int(splits[1])+1
            end_frame = int(splits[2])+1
            sample_frames = range(start_frame, end_frame, sample_rate)
            video_name = splits[0]
            sampled_imgs = [(video_name, sample_frame) for sample_frame in sample_frames]
            all_imgs += sampled_imgs
    return all_imgs


class YouCook_dataset(data.Dataset):
    def __init__(self, data_root, img_infos, transform_ops=None):
        self.transform_ops = transform_ops
        self.to_tensor = transforms.ToTensor()
        self.data_root = data_root
        self.img_infos = img_infos

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, index):
        (video_name, img_index) = self.img_infos[index]
        dst_name = '{}_{}.pkl'.format(video_name, str(img_index).zfill(6))
        img_path = os.path.join(self.data_root, video_name, '{}.jpg'.format(str(img_index).zfill(6)))
        img = Image.open(img_path)
        if self.transform_ops is not None:
            img = self.transform_ops(img)
        else:
            img = self.to_tensor(img)
        return (img, dst_name)


def check_available(img_infos):
    missing_videos = set()
    data_root = '/home/liangkeg/main_storage/data/youcookii/raw_frames'
    for img_info in img_infos:
        video_name, img_id = img_info[0], img_info[1]
        frame_path = os.path.join(data_root, video_name, '{}.jpg'.format(str(img_id).zfill(6)))
        if not os.path.exists(frame_path):
            missing_videos.add(video_name)
            print(frame_path)
        else:
            print(frame_path)
    return missing_videos

if __name__ == '__main__':
    resnet152 = models.resnet152(pretrained=True)
    modules = list(resnet152.children())[:-1]
    resnet152 = nn.Sequential(*modules)
    for p in resnet152.parameters():
        p.requires_grad = False

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet152.to(device)
    resnet152.eval()

    dst_root = '/home/liangkeg/main_storage/data/youcookii/features/raw_resnet_features'
    data_root = '/home/liangkeg/main_storage/data/youcookii/raw_frames'
    split_file = '/home/liangkeg/main_storage/data/youcookii/data_splits/all_data.lst'
    img_infos = load_imgs(split_file)
    # missing_videos = check_available(img_infos)
    # prrnt(missing_videos)
    # assert False
    all_data = YouCook_dataset(data_root, img_infos, transform_ops = data_transforms['val'])
    data_loader = DataLoader(all_data, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

    for ind_batch, (imgs, img_infos) in enumerate(data_loader):
        print('Process {}th of {} batches'.format(ind_batch, len(data_loader)))
        input_data = imgs.to(device)
        features = resnet152(input_data)
        cpu_features = features.cpu().numpy()
        num_samples = cpu_features.shape[0]
        for ind_img in range(num_samples):
            feature = cpu_features[ind_img,:]
            dst_name = img_infos[ind_img]
            # print(np.squeeze(feature).shape, dst_name)
            pickle.dump(feature, open(os.path.join(dst_root, dst_name), 'wb'))
            # assert False

# {'bjjIgdnB1Y', 'wHDZkh-21G0'}
