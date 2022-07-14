from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class HMDataset(Dataset):
    
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
