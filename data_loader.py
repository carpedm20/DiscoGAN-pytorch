import os
from glob import glob
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

PIX2PIX_DATASETS = ['facades', 'edges2shoes', 'edges2handbags']

def pix2pix_split_images(paths):
    for path in paths:
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.name = os.path.basename(root)

        if self.name in PIX2PIX_DATASETS:
            pass

        self.a_paths = glob(os.path.join(self.root, 'A/*'))
        self.b_paths = glob(os.path.join(self.root, 'B/*'))

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, []

    def __len__(self):
        return len(self.paths)

def get_loader(root, transform, batch_size, shuffle=True, num_workers=2):
    dataset = Dataset(root=root, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
