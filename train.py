"""Trains the LSTM autoencoder (normal data only) and the GNN (graph snapshots)."""
import numpy as np, torch, torch.nn as nn, pickle, json
from torch_geometric.loader import DataLoader
from common import *

torch.manual_seed(0); np.random.seed(0)
tel, edges = load()
base = fit_baseline(tel)
z = zscores(tel, base)

# ---------- LSTM: learn normal behaviour only ----------
seqs = []
for h in HOSTS:
    m = (tel.host == h).values
    zh, lab, t = z[m], tel.label.values[m], tel.t.values[m]
    S = make_sequences(zh)
    ok = [(t[i + SEQ_LEN - 1] < TRAIN_END) and lab[i:i + SEQ_LEN].sum() == 0
          for i in range(len(S))]
    seqs.append(S[np.array(ok)])
X = torch.tensor(np.concatenate(seqs))
print("LSTM training sequences (normal only):", len(X))
lstm = LSTMAutoencoder(); opt = torch.optim.Adam(lstm.parameters(), 1e-3)
log = {'lstm': [], 'gnn': []}               # per-epoch loss, saved for the report
for ep in range(25):
    perm = torch.randperm(len(X)); tot = 0
    for i in range(0, len(X), 128):
        b = X[perm[i:i + 128]]
        loss = nn.functional.mse_loss(lstm(b), b)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
    log['lstm'].append(tot / len(X))
    if ep % 6 == 0 or ep == 24: print(f"  LSTM epoch {ep+1:2d}  loss {tot/len(X):.4f}")
# scale for converting errors into a 0-100 risk (99th percentile of normal errors)
errs = []
for h in HOSTS:                                   # normal-only windows, so attacks don't inflate the scale
    m = (tel.host == h).values
    e = lstm_errors(lstm, z[m]); lab = tel.label.values[m]
    for i in range(SEQ_LEN - 1, TRAIN_END):
        if lab[i - SEQ_LEN + 1:i + 1].sum() == 0: errs.append(e[i])
errs = np.array(errs)
q99 = float(np.nanquantile(errs, 0.99))

# ---------- GNN: learn communication structure ----------
graphs = build_graphs(tel, edges, z, range(0, TRAIN_END))
loader = DataLoader(graphs, batch_size=64, shuffle=True)
gnn = GNN(); opt = torch.optim.Adam(gnn.parameters(), 5e-3)
w = torch.tensor([1.0, 8.0])                         # attacks are rare -> weight them up
for ep in range(30):
    tot = 0
    for b in loader:
        loss = nn.functional.cross_entropy(gnn(b), b.y, weight=w)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    log['gnn'].append(tot / len(loader))
    if ep % 10 == 0 or ep == 29: print(f"  GNN  epoch {ep+1:2d}  loss {tot/len(loader):.4f}")

torch.save({"lstm": lstm.state_dict(), "gnn": gnn.state_dict()}, "outputs/models.pt")
pickle.dump({"base": base, "q99": q99}, open("outputs/scalers.pkl", "wb"))
log.update(lstm_sequences=len(X), gnn_graphs=len(graphs), q99=q99)
json.dump(log, open("outputs/training_log.json", "w"))
print("saved outputs/models.pt and outputs/scalers.pkl  (LSTM q99 error = %.4f)" % q99)
