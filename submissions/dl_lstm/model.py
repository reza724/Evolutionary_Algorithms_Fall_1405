"""DL sample: tiny LSTM over the last 12 bars predicts the TP/SL label class; LabelModel trades it."""
import numpy as np
import torch
from torch import nn

import bench

FEATS = ["ret_1", "range", "qvol_chg", "taker_ratio", "prem"]
N_TRAIN = 60_000                               # last sequences of the train file: keeps the sample at ~1 min on a CPU
torch.set_num_threads(2)


class Net(nn.Module):
    def __init__(self, n_classes, n_in=len(FEATS), hidden=16):
        super().__init__()
        self.lstm, self.head = nn.LSTM(n_in, hidden, batch_first=True), nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1])


class LSTM(bench.LabelModel):
    model_type, author = "dl", "sample"
    lookback = 12                              # the sequence length
    horizon, levels = 12, [1, 2]

    def fit(self, train):
        ok, y = self.targets(train)
        f = np.nan_to_num(train[FEATS].to_numpy(dtype=np.float32))
        self.mu, self.sd = f.mean(0), f.std(0) + 1e-8
        ends = np.flatnonzero(ok)                                   # a sequence ending at bar b gets the label of bar b
        keep = ends >= self.lookback - 1
        ends, y = ends[keep][-N_TRAIN:], y[keep][-N_TRAIN:]
        X = torch.tensor(np.stack([(f[b - self.lookback + 1: b + 1] - self.mu) / self.sd for b in ends]))
        y = torch.tensor(y, dtype=torch.int64)

        self.net, loss_fn = Net(2 * len(self.levels) + 2), nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        for _ in range(3):
            for idx in torch.randperm(len(X)).split(512):
                opt.zero_grad()
                loss = loss_fn(self.net(X[idx]), y[idx])
                loss.backward()
                opt.step()
        self.net.eval()
        with torch.no_grad():
            acc = (self.net(X).argmax(1) == y).float().mean()
        print(f"dl_lstm: {len(X):,} sequences, last batch CE {loss.item():.3f}, train acc {acc:.3f}")

    def predict(self, obs):
        x = (np.nan_to_num(obs.history[FEATS].to_numpy(dtype=np.float32)) - self.mu) / self.sd
        with torch.no_grad():
            return int(self.net(torch.tensor(x[None])).argmax(1))
