import io
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image
import pytorch_lightning as pl
from typing import Optional
from torchvision import transforms
from google.cloud import storage


class LocalHMDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = Path.cwd() / img_dir
        self.img_names = self._get_img_names()
        self.transform = transform
        
    def _get_img_names(self):
        return [img_name.name for img_name in self.img_dir.glob('*.jpg')]

    def __getitem__(self, idx):
        img_path = self.img_dir / self.img_names[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        article_id = img_path.name
        return img, article_id

    def __len__(self):
        return len(self.img_names)


class GcpHMDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.bucket_name = img_dir.split('/')[0]
        self.data_folder = img_dir.split('/', maxsplit=1)[-1]
        self.storage_client = storage.Client()
        self.transform = transform
        self.img_names = self._get_img_names()
        
    def _get_img_names(self):
        blobs = self.storage_client.list_blobs(self.bucket_name, prefix=self.data_folder)
        img_names = [blob.name.split('/')[-1] for blob in blobs if blob.name[-3:]=='jpg']
        return img_names

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = Path(self.data_folder, img_name)

        bucket = self.storage_client.get_bucket(self.bucket_name)
        blob = storage.Blob(str(img_path), bucket)
        blob = blob.download_as_string()
        bytes_object = io.BytesIO(blob)
        img = Image.open(bytes_object).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        article_id = img_path.name
        return img, article_id

    def __len__(self):
        return len(self.img_names)


class HMDataLoader(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int=32, num_workers: int=0, platform: str='local'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self._platform = platform
        self.pl_dataset = {'local': LocalHMDataset, 'gcp': GcpHMDataset}.get(self._platform)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            hm_dataset = self.pl_dataset(self.data_dir, transform=self.transform)
            train_size, val_size = self._get_train_valid_size(hm_dataset, frac=0.9)
            self.hm_train, self.hm_val = random_split(hm_dataset, [train_size, val_size])
        elif stage == 'predict' or stage is None:
            self.hm_predict = self.pl_dataset(self.data_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.hm_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.hm_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.hm_predict, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    # def teardown(self, stage: Optional[str] = None):
    #    pass

    @staticmethod
    def _get_train_valid_size(dataset, frac=0.9):
        data_size = len(dataset)
        train_size = int(frac*data_size)
        val_size = data_size - train_size
        return [train_size, val_size]
