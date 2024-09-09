import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class VocosDataModule(LightningDataModule):
    def __init__(self, train_filelist, val_filelist, batch_size, num_workers):
        super().__init__()
        self.train_filelist = train_filelist
        self.val_filelist = val_filelist
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _get_dataloader(self, filelist, train):
        dataset = ReflowDataset(filelist)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=train,
            num_workers=self.num_workers, pin_memory=True)
        return dataloader

    def train_dataloader(self):
        return self._get_dataloader(self.train_filelist, train=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_filelist, train=False)


class ReflowDataset(Dataset):
    def __init__(self, filelist):
        super().__init__()
        with open(filelist) as f:
            self.filelist = f.read().splitlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        th_path = self.filelist[idx]
        return torch.load(th_path)
