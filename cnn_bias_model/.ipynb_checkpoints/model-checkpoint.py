from torch import nn


class BiasNet(nn.Module):
    def __init__(self, seq_len=128, n_filters=32, kernel_size=5) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.conv1 = nn.Sequential(
            # 4 is for the 4 nucleotides
            nn.Conv1d(4, self.n_filters, kernel_size=self.kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(self.n_filters, self.n_filters, kernel_size=self.kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(928, 1024),
            nn.Linear(1344, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
            nn.Linear(1024, self.seq_len),
        )
        
        self.output = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        x = self.output(x)

        return x


if __name__ == "__main__":
    model = BiasNet()

    import torch
    from utils import random_seq, one_hot_encode

    seq1 = random_seq(181)
    seq2 = random_seq(181)
    x1 = one_hot_encode(seq1)
    x1 = torch.tensor(x1).float()

    x2 = one_hot_encode(seq2)
    x2 = torch.tensor(x2).float()

    x1 = x1.unsqueeze(dim=0)
    x2 = x2.unsqueeze(dim=0)

    x = torch.cat((x1, x2), dim=0)

    x = model(x)

    print(x)
