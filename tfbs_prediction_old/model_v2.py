from collections.abc import Sequence
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq_encoder import SeqEncoder

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class AttentionBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        n_embd: int,
        nhead: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, nhead, batch_first=True)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlpf = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.Linear(4 * n_embd, n_embd),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def _sa_block(self, x, key_padding_mask=None, attn_mask=None):
        x, w = self.attn(
            x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        return x, w

    def forward(self, x):
        att_x, att_w = self._sa_block(self.ln_1(x))
        x = x + att_x
        x = x + self.mlpf(self.ln_2(x))
        return x, att_w
    

class ACCESSNet(nn.Module):
    def __init__(
        self,
        peak_len: int,
        n_filters: Sequence[int] | None = None,
        n_channels: int = 4,
        kernel_size: int = 5,
        n_dims: int = 16,
        dropout_rate: float = 0.25,
    ) -> None:
        if n_filters is None:
            n_filters = [64, 32, 32, 16]
        super().__init__()

        self.peak_len = peak_len

        # parameters for sequence encoder
        self.n_filters = n_filters
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_dims = n_dims
        self.dropout_rate = dropout_rate
        self.n_attn_blocks = 1

        # build sequence encoders
        self.seq_encoder = SeqEncoder(
            base_size=self.n_channels,
            kernel_size=self.kernel_size,
            n_filters=self.n_filters,
        )
        self.embd_len = (self.peak_len // (2 ** len(self.n_filters))) * self.n_dims

        self.atac_signal_embd = nn.Linear(self.peak_len, self.embd_len)
        self.atac_bias_embd = nn.Linear(self.peak_len, self.embd_len)
        self.access_signal_embd = nn.Linear(self.peak_len, self.embd_len)
        self.access_bias_embd = nn.Linear(self.peak_len, self.embd_len)

        # fully connected layers to predict TF binding site
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.embd_len, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, peak_seq, atac_signal=None, atac_bias=None, access_signal=None, access_bias=None):
        # Embed peak sequence
        seq_embd = self.seq_encoder(peak_seq)  # (batch, 1, L, 4)
        seq_embd = torch.flatten(seq_embd.permute(0, 2, 1, 3), start_dim=2)

        # Embed ATAC and ACCESS signals
        if atac_signal is not None and atac_bias is not None:
            atac_signal = self.atac_signal_embd(atac_signal)
            atac_bias = self.atac_bias_embd(atac_bias)
            seq_embd = seq_embd + atac_signal + atac_bias

        if access_signal is not None and access_bias is not None:
            access_signal = self.access_signal_embd(access_signal)
            access_bias = self.access_bias_embd(access_bias)
            seq_embd = seq_embd + access_signal + access_bias

        x = self.fc(seq_embd)

        return x
    

if __name__ == "__main__":
    # unit test
    batch_size = 20
    peak_len = 256

    model = ACCESSNet(
        peak_len=peak_len,
        n_filters=[64, 32, 32, 16],
        n_channels=4,
        kernel_size=5,
        n_dims=16,
        dropout_rate=0.25,
    )

    peak_seq = torch.randn(batch_size, 1, peak_len, 4) # (batch_size, 1, peak_len, n_channels)
    atac_signal = torch.randn(batch_size, 1, peak_len) # (batch_size peak_len)
    atac_bias = torch.randn(batch_size, 1, peak_len) # (batch_size, peak_len)

    output = model(peak_seq, atac_signal=atac_signal, atac_bias=atac_bias)
    print(output)
