import os.path as osp
import torch.utils.data as data
import basicsr.data.util as util
import os
from basicsr.data.transforms import paired_random_crop, random_augmentation


class Dataset_SMIDImage(data.Dataset):
    def __init__(self, opt):
        super(Dataset_SMIDImage, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.io_backend_opt = opt['io_backend']
        self.data_type = self.io_backend_opt['type']
        self.data_info = {'path_LQ': [], 'path_GT': [],
                          'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        # Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}

        testing_dir = []
        data_root = os.path.dirname(self.GT_root)

        f = open(os.path.join(data_root, 'test_list.txt'))
        lines = f.readlines()
        for mm in range(len(lines)):
            this_line = lines[mm].strip()
            testing_dir.append(this_line)

        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)

        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            subfolder_name = osp.basename(subfolder_LQ)

            if self.opt['phase'] == 'train':
                if (subfolder_name in testing_dir):
                    continue
            else:  # val, test
                if not (subfolder_name in testing_dir):
                    continue

            img_paths_LQ = util.glob_file_list(subfolder_LQ)
            img_paths_GT_all = util.glob_file_list(subfolder_GT)
            img_paths_GT = []
            for mm in range(len(img_paths_GT_all)):
                if '.ARW' in img_paths_GT_all[mm] or 'half' in img_paths_GT_all[mm]:
                    continue
                img_paths_GT.append(img_paths_GT_all[mm])

            max_idx = len(img_paths_LQ)
            self.data_info['path_LQ'].extend(
                img_paths_LQ)  # list of path str of images
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append('{}/{}'.format(i, max_idx))

            border_l = [0] * max_idx
            for i in range(self.half_N_frames):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            if self.cache_data:
                self.imgs_LQ[subfolder_name] = img_paths_LQ
                self.imgs_GT[subfolder_name] = img_paths_GT

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        img_LQ_path = self.imgs_LQ[folder][idx:idx + 1]
        img_GT_path = self.imgs_GT[folder][0]
        img_GT_path = [img_GT_path]

        img_LQ = util.read_img_seq_smid(img_LQ_path)
        img_LQ = img_LQ[0]
        img_GT = util.read_img_seq_smid(img_GT_path)
        img_GT = img_GT[0]

        if self.opt['phase'] == 'train':

            if self.opt['train_size'] < 512:
                img_GT, img_LQ = paired_random_crop(img_GT, img_LQ, self.opt['train_size'], 1, img_GT_path)

            if self.geometric_augs:
                img_GT, img_LQ = random_augmentation(img_GT, img_LQ)
            
            img_GT, img_LQ = util.smid_img_to_tensor([img_GT, img_LQ])
        else:
            img_GT, img_LQ = util.smid_img_to_tensor([img_GT, img_LQ])

        return {
            'lq': img_LQ,
            'gt': img_GT,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border,
            'lq_path': img_LQ_path[0],
            'gt_path': img_GT_path[0]
        }

    def __len__(self):
        return len(self.data_info['path_LQ'])
