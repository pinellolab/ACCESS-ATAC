import torch
from typing import List, Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader


class DNASeqDataSet(Dataset):
    def __init__(self, x: np.array, y: np.array = None, train: bool = True) -> None:
        super().__init__()

        self.x = x
        self.y = y
        self.train = train
        self.len = x.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.train:
            return (self.x[index], self.y[index])

        else:
            return self.x[index]


def get_dataloader(
    x: np.array,
    y: Optional[np.array] = None,
    batch_size: int = 64,
    num_workers: int = 2,
    drop_last: bool = False,
    shuffle: bool = True,
    train: bool = True,
):
    dataset = DNASeqDataSet(x=x, y=y, train=train)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        drop_last=drop_last,
        persistent_workers=True,
    )

    return dataloader
