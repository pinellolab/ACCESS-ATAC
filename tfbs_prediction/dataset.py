import torch
from typing import Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ChromatinAccessibilityDataSet(Dataset):
    def __init__(self, 
                 seq: np.array = None, 
                 label: np.array = None,
                 atac_signal: np.array = None,
                 atac_bias: np.array = None,
                 access_signal: np.array = None,
                 access_bias: np.array = None):
        super().__init__()
        
        # make sure all arrays have the same length
        assert seq.shape[0] == atac_signal.shape[0] == atac_bias.shape[0] == access_signal.shape[0] == access_bias.shape[0] == label.shape[0]
        
        self.seq = torch.tensor(seq, dtype=torch.float)
        self.seq = self.seq.unsqueeze(1)

        self.atac_signal = torch.tensor(atac_signal, dtype=torch.float).unsqueeze(1)
        self.atac_bias = torch.tensor(atac_bias, dtype=torch.float).unsqueeze(1)
        self.access_signal = torch.tensor(access_signal, dtype=torch.float).unsqueeze(1)
        self.access_bias = torch.tensor(access_bias, dtype=torch.float).unsqueeze(1)
        self.label = label
        self.len = seq.shape[0]
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = {}
        data['seq'] = self.seq[index]
        data['atac_signal'] = self.atac_signal[index]
        data['atac_bias'] = self.atac_bias[index]
        data['access_signal'] = self.access_signal[index]
        data['access_bias'] = self.access_bias[index]
        data['label'] = self.label[index]

        return data



def get_dataloader(
    seq: np.array,
    atac_signal: np.array = None,
    atac_bias: np.array = None,
    access_signal: np.array = None,
    access_bias: np.array = None,
    label: Optional[np.array] = None,
    batch_size: int = 64,
    num_workers: int = 1,
    drop_last: bool = False,
    shuffle: bool = True,
):
    dataset = ChromatinAccessibilityDataSet(seq=seq, 
                                            atac_signal=atac_signal, 
                                            atac_bias=atac_bias, 
                                            access_signal=access_signal, 
                                            access_bias=access_bias, 
                                            label=label)

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
