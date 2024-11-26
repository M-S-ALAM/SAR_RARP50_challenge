
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from functools import lru_cache

class EfficientCSVImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, mask_dir=None, transform=None, cache_size=1000):
        self.data_frame = pd.read_csv(csv_file)[:100]
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.cache_image = lru_cache(maxsize=cache_size)(self._load_image)
        self.cache_mask = lru_cache(maxsize=cache_size)(self._load_mask) if mask_dir else None

    def __len__(self):
        return len(self.data_frame)

    def _load_image(self, path):
        return Image.open(path).convert('RGB')

    def _load_mask(self, path):
        return Image.open(path).convert('L')

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = self.cache_image(img_name)
        mask = self.cache_mask(os.path.join(self.mask_dir, self.data_frame.iloc[idx, 0].replace('.jpg', '_mask.png'))) if self.mask_dir else None
        if self.transform:
            image = self.transform(image)
            if mask is not None:
                mask = self.transform(mask)
        return image, mask if mask is not None else (image, None)

# Data transformations
transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create data loaders
def create_data_loaders(train_csv_file, test_csv_file, batch_size=8):
    train_dataset = EfficientCSVImageDataset(train_csv_file, '.', '.', transformations, cache_size=1000)
    test_dataset = EfficientCSVImageDataset(test_csv_file, '.', '.', transformations, cache_size=1000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return train_loader, test_loader
