import os
import cv2
import torch
from torch.utils.data import Dataset


class PlantTraitsDataset(Dataset):
    def __init__(self, df, class_names, feature_names, base_path='./planttraits2024', split='train',
                 transforms=None) -> None:
        super().__init__()
        self.split = split
        self.imgs_path = os.path.join(base_path, f"{split}_images")
        self.class_names = class_names
        self.feature_names = feature_names
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        curr = self.df.iloc[index]

        file_path = os.path.join(self.imgs_path, f"{int(curr['id'])}.jpeg")
        img = cv2.imread(file_path)
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float() / 255
        features = torch.from_numpy(curr[self.feature_names].values).float()

        if self.transforms is not None:
            img = self.transforms(img.permute(2, 0, 1))

        sample = {}
        sample['img'] = img
        sample['features'] = features
        sample['id'] = curr['id']

        if True or self.split == 'train':
            labels = torch.from_numpy(curr[self.class_names].values).float()
            sample['labels'] = labels

        return sample
