import torch
from typing import Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ChromatinAccessibilityDataSet(Dataset):
    def __init__(self, 
                 seq: np.ndarray, 
                 label: np.ndarray,
                 signal_accessatac_tn5: np.ndarray,
                 signal_accessatac_dddss: np.ndarray):
        super().__init__()
        
        # make sure all arrays have the same length
        # print(seq.shape)
        # print(label.shape)
        # print(signal_accessatac_tn5.shape)
        # print(signal_accessatac_dddss.shape)

        assert seq.shape[0] == signal_accessatac_tn5.shape[0] == label.shape[0] == signal_accessatac_dddss.shape[0]
        
        self.seq = torch.tensor(seq, dtype=torch.float)
        self.seq = self.seq.unsqueeze(1)

        self.signal_accessatac_tn5 = torch.tensor(signal_accessatac_tn5, dtype=torch.float).unsqueeze(1)
        self.signal_accessatac_dddss = torch.tensor(signal_accessatac_dddss, dtype=torch.float).unsqueeze(1)

        self.label = label
        self.len = seq.shape[0]
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = {}
        data['seq'] = self.seq[index]
        data['signal_accessatac_tn5'] = self.signal_accessatac_tn5[index]
        data['signal_accessatac_dddss'] = self.signal_accessatac_dddss[index]
        data['label'] = self.label[index]

        return data


def get_dataloader(
    seq: np.ndarray,
    signal_accessatac_tn5: np.ndarray,
    signal_accessatac_dddss: np.ndarray,
    label: Optional[np.ndarray] = None,
    batch_size: int = 64,
    num_workers: int = 1,
    drop_last: bool = False,
    shuffle: bool = True,
):
    dataset = ChromatinAccessibilityDataSet(seq=seq, 
                                            signal_accessatac_tn5=signal_accessatac_tn5,
                                            signal_accessatac_dddss=signal_accessatac_dddss,
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
