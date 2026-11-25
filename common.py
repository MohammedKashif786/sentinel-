"""Feature preparation + the two neural models (LSTM autoencoder, GraphSAGE GNN)."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from config import *

def load():
    tel = pd.read_csv("data/telemetry.csv")
    edges = pd.read_csv("data/edges.csv")
    return tel, edges

def fit_baseline(tel):
    """Per-host 'what is normal' statistics, learnt from normal training windows only."""
    base = {}
    train = tel[(tel.t < TRAIN_END) & (tel.label == 0)]
    for h, g in train.groupby("host"):
        x = np.log1p(g[FEATURES].values)
        base[h] = (x.mean(0), x.std(0) + 1e-3)
    return base

def zscores(tel, base):
    """Personalised z-score of every feature for every host and window."""
    z = np.zeros((len(tel), len(FEATURES)))
    for h, (mu, sd) in base.items():
        m = (tel.host == h).values
        z[m] = np.clip((np.log1p(tel.loc[m, FEATURES].values) - mu) / sd, -10, 10)
    return z

# ---------------------------------------------------------------- LSTM
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_feat=len(FEATURES), hid=32):
        super().__init__()
        self.enc = nn.LSTM(n_feat, hid, batch_first=True)
        self.dec = nn.LSTM(hid, hid, batch_first=True)
        self.out = nn.Linear(hid, n_feat)
    def forward(self, x):
        _, (h, _) = self.enc(x)
        rep = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        d, _ = self.dec(rep)
        return self.out(d)

def make_sequences(zhost):
    """zhost: (T, F) for one host -> (T-L+1, L, F) sliding windows."""
    L = SEQ_LEN
    return np.stack([zhost[i:i + L] for i in range(len(zhost) - L + 1)]).astype("float32")

def lstm_errors(model, zhost):
    """Reconstruction error of the most recent 3 windows in each sequence."""
    seq = torch.tensor(make_sequences(zhost))
    with torch.no_grad():
        rec = model(seq)
    err = ((rec - seq) ** 2)[:, -3:, :].mean((1, 2)).numpy()
    return np.concatenate([np.full(SEQ_LEN - 1, np.nan), err])

# ---------------------------------------------------------------- GNN
class GNN(nn.Module):
    def __init__(self, n_in=len(FEATURES) + 1, hid=32):
        super().__init__()
        self.c1, self.c2 = SAGEConv(n_in, hid), SAGEConv(hid, hid)
        self.fc = nn.Linear(hid, 2)
    def forward(self, d):
        x = torch.relu(self.c1(d.x, d.edge_index))
        x = torch.relu(self.c2(x, d.edge_index))
        return self.fc(x)

def build_graphs(tel, edges, z, t_range):
    """One graph per time window: nodes = 5 hosts + server, edges = observed communication."""
    idx = {n: i for i, n in enumerate(NODES)}
    tel = tel.reset_index(drop=True)
    zdf = pd.DataFrame(z, columns=FEATURES); zdf["t"] = tel.t; zdf["host"] = tel.host
    zdf["label"] = tel.label
    gby_z = dict(tuple(zdf.groupby("t")))
    gby_e = dict(tuple(edges.groupby("t")))
    graphs = []
    for t in t_range:
        zz = gby_z[t].set_index("host")
        x = np.zeros((len(NODES), len(FEATURES) + 1), dtype="float32")
        y = np.zeros(len(NODES), dtype="int64")
        for h in HOSTS:
            x[idx[h], :-1] = zz.loc[h, FEATURES].values
            y[idx[h]] = int(zz.loc[h, "label"])
        x[idx["server"], -1] = 1.0                      # marks the server node
        e = gby_e[t]; e = e[e.flows >= 2]               # ignore trivial one-off contacts
        src = [idx[s] for s in e.src] + [idx[d] for d in e.dst]
        dst = [idx[d] for d in e.dst] + [idx[s] for s in e.src]
        graphs.append(Data(x=torch.tensor(x), y=torch.tensor(y),
                           edge_index=torch.tensor([src, dst], dtype=torch.long), t=t))
    return graphs
