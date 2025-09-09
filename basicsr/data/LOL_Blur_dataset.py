from torch.utils import data as data
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
import torchvision.transforms.functional as TF
from typing import Sequence

from PIL import Image

import random
import torch
import os

class RandomRotation(torch.nn.Module):
    def __init__(self, angles: Sequence[int]):
        super().__init__()
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

class Dataset_LOLBlurImage(data.Dataset):
    def __init__(self, opt):
        super(Dataset_LOLBlurImage, self).__init__()
        self.opt = opt
        gt_folder, lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.lq_paths, self.gt_paths = self._get_imgs_path(lq_folder, gt_folder)
        self.to_tensor = ToTensor()

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']
            self.random_crop = RandomCrop([opt['gt_size'], opt['gt_size']])
            if self.geometric_augs:
                self.transform = Compose([
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomRotation([0, 90 ,180, 270])
                ])

    def _flatten_list_comprehension(self, matrix):
        return [item for row in matrix for item in row]
    

    def _check_paths(self, list_of_lists):
        '''
        check if all the image routes are correct
        '''
        paths = self._flatten_list_comprehension(list_of_lists)
        trues = [os.path.isfile(file) for file in paths]
        counter = 0
        for true, path in zip(trues, paths):
            if true != True:
                print('Non valid route!', path)
                counter +=1

    def _get_imgs_path(self, lq_path, gt_path):
        """Get image paths from the root path."""
        lq_paths = [os.path.join(lq_path, path) for path in os.listdir(lq_path)]
        gt_paths = [os.path.join(gt_path, path) for path in os.listdir(gt_path)]        
        
        lq_paths = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in lq_paths ]
        gt_paths = [[os.path.join(path_element, path_png) for path_png in os.listdir(path_element)] for path_element in gt_paths ]
        lq_paths = self._flatten_list_comprehension(lq_paths)
        gt_paths = self._flatten_list_comprehension(gt_paths)

        return lq_paths, gt_paths

    def __getitem__(self, index):
        index = index % len(self.lq_paths)
        gt_path = self.gt_paths[index]
        img_gt = self.to_tensor(Image.open(gt_path).convert('RGB'))

        lq_path = self.lq_paths[index]
        img_lq = self.to_tensor(Image.open(lq_path).convert('RGB'))

        if self.opt['phase'] == 'train':
            high_and_low = torch.stack((img_gt, img_lq))
            high_and_low = self.random_crop(high_and_low)
            if self.geometric_augs:
                high_and_low = self.transform(high_and_low)
            img_gt, img_lq = high_and_low

        return {
            'lq': img_lq.float(),
            'gt': img_gt.float(),
            'lq_path': lq_path,
            'gt_path': gt_path
        }
        

    def __len__(self):
        return len(self.gt_paths)