import torch
from typing import List, Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ChromatinAccessibilityDataSet(Dataset):
    def __init__(self, 
                 seq: np.array, 
                 y: np.array,
                 atac_signal: np.array = None,
                 atac_bias: np.array = None,
                 access_signal: np.array = None,
                 access_bias: np.array = None) -> None:
        super().__init__()
        
        # make sure all arrays have the same length
        assert seq.shape[0] == atac_signal.shape[0] == atac_bias.shape[0] == access_signal.shape[0] == access_bias.shape[0] == y.shape[0]

        self.seq = seq
        self.atac_signal = atac_signal
        self.atac_bias = atac_bias
        self.access_signal = access_signal
        self.access_bias = access_bias
        self.y = y
        self.len = seq.shape[0]
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.seq[index], self.atac_signal[index], self.atac_bias[index], self.access_signal[index], self.access_bias[index], self.y[index])



def get_dataloader(
    x: np.array,
    y: Optional[np.array] = None,
    batch_size: int = 64,
    num_workers: int = 1,
    drop_last: bool = False,
    shuffle: bool = True,
    train: bool = True,
):
    dataset = ChromatinAccessibilityDataSet(x=x, y=y, train=train)

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
