import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation

        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
        )

        self.out = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.out(x + self.block(x))
    

class MaxATACCNN(nn.Module):
    def __init__(
        self,
        in_ch: int = 5,
        n_filters: int = 15,
        n_blocks: int = 10,
        dil_start: int = 1,
        dil_mult: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.1,
        head_pool: str = "mean",  # "mean" or "attn"
    ) -> None:
        super().__init__()

        self.in_ch = in_ch
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.head_pool = head_pool

        # stem
        self.stem = nn.Sequential(
            nn.Conv1d(self.in_ch, self.n_filters, kernel_size=15, padding=7),
            nn.ReLU(inplace=True),
        )

        # trunk (dilated residual stack)
        blocks = []
        dil = dil_start
        for _ in range(self.n_blocks):
            blocks.append(ResidualBlock(self.n_filters, 
                                        kernel_size=self.kernel_size, 
                                        dilation=dil, dropout=self.dropout))
            dil = min(512, dil * dil_mult)
        self.trunk = nn.Sequential(*blocks)

        # pooling / attention head
        if self.head_pool == "mean":
            self.pool = nn.AdaptiveAvgPool1d(1)  # (B,C,L)->(B,C,1)
            self.attn = None
        else:
            self.pool = None
            self.attn = nn.Sequential(
                nn.Conv1d(self.n_filters, 1, kernel_size=1),
                nn.Softmax(dim=-1),
            )

        # classifier
        self.fc = nn.Sequential(
            nn.Flatten(),                 # (B,C,1) -> (B,C)
            nn.Linear(self.n_filters, 1),   # logits
        )

    def forward(self, x):
        # expect x: (B, L, C_in) -> permute to (B, C_in, L)
        x = x.permute(0, 2, 1)

        x = self.stem(x)
        x = self.trunk(x)

        if self.head_pool == "mean":
            x = self.pool(x)                    # (B,C,1)
        else:
            w = self.attn(x)                    # (B,1,L)
            x = (x * w).sum(dim=-1, keepdim=True)  # (B,C,1)

        x = self.fc(x)                          # (B,1)                   # logits
        return x.squeeze(-1)

if __name__ == "__main__":
    
    import torch
    import numpy as np
    from utils import random_seq, one_hot_encode

    seq1 = random_seq(256)
    seq2 = random_seq(256)
    x1 = one_hot_encode(seq1)
    x2 = one_hot_encode(seq2)

    signal_raw = np.expand_dims(np.random.rand(256), axis=1)
    x1 = np.concatenate([x1, signal_raw], axis=1)
    x2 = np.concatenate([x2, signal_raw], axis=1)
    
    x1 = torch.tensor(x1).float()
    x2 = torch.tensor(x2).float()

    # # add ATAC-seq signal
    x1 = x1.unsqueeze(dim=0)
    x2 = x2.unsqueeze(dim=0)

    x = torch.cat((x1, x2), dim=0)

    print(x.shape)

    model = MaxATACCNN(in_ch=5)

    x = model(x)

    print(x)
