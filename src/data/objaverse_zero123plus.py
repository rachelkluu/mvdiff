import os
import json
import numpy as np
import webdataset as wds
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from pathlib import Path

from src.utils.train_util import instantiate_from_config


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):

        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=4, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='20240924_3D Dataset',
        validation=False,
    ):
        self.root_dir = Path(root_dir)
        self.paths = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])

        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[-16:]  # use last 16 folders for validation
        else:
            self.paths = self.paths[:-16]  # use the rest for training
        print(f'============= length of dataset {len(self.paths)} =============')

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        pil_img = Image.open(path)
        image = np.asarray(pil_img, dtype=np.float32) / 255.
        image = image[:, :, :3]  # Remove alpha channel (RGBA → RGB)
        return torch.from_numpy(image).permute(2, 0, 1).contiguous().float()


    def __getitem__(self, index):
        folder_path = os.path.join(self.root_dir, self.paths[index])

        # Load query image (conditional image)
        cond_img_path = os.path.join(folder_path, 'query_image.png')
        bkg_color = [1., 1., 1.]  # white background color

        img_list = []
        try:
            cond_img = self.load_im(cond_img_path, bkg_color)
            for i in range(1, 7):
                target_img_path = os.path.join(folder_path, f'target_image_{i}.png')
                target_img = self.load_im(target_img_path, bkg_color)
                img_list.append(target_img)
        except Exception as e:
            print(f"Error loading images in folder {folder_path}: {e}")
            index = np.random.randint(0, len(self.paths))
            return self.__getitem__(index)

        imgs = torch.stack(img_list, dim=0).float()
        data = {
            'cond_imgs': cond_img,      # (3, H, W)
            'target_imgs': imgs,        # (6, 3, H, W)
        }
        return data
