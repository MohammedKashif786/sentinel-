"""Figures for the report: preprocessing steps and training evidence."""
import json, pickle, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from common import *
import os; os.makedirs("outputs/figures", exist_ok=True)
tel, edges = load(); base = fit_baseline(tel); z = zscores(tel, base)
tel = tel.reset_index(drop=True); log = json.load(open("outputs/training_log.json"))
res = pd.read_csv("outputs/results.csv")

# ---------- Figure 1: preprocessing of one feature for one host
h = "h4"; m = (tel.host == h).values; sub = tel[m]; fi = FEATURES.index("conn_freq")
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
ax[0].plot(sub.t, sub.conn_freq, color="#1F3864"); ax[0].set_title("1. Raw telemetry\n(connections / min, h4)")
ax[1].plot(sub.t, np.log1p(sub.conn_freq), color="#8b5cf6"); ax[1].set_title("2. Log transform\n(tames the heavy tail)")
ax[2].plot(sub.t, z[m][:, fi], color="#0891b2"); ax[2].axhline(0, color="#999", lw=.8)
ax[2].set_title("3. Per-device z-score\n(distance from h4's own normal)")
seq = make_sequences(z[m])[2400 - SEQ_LEN + 60]
ax[3].imshow(seq.T, aspect="auto", cmap="coolwarm", vmin=-6, vmax=6)
ax[3].set_yticks(range(len(FEATURES))); ax[3].set_yticklabels(FEATURES, fontsize=7)
ax[3].set_xlabel("12 windows (2 minutes)"); ax[3].set_title("4. One LSTM input sequence\n(12 windows x 10 features, attack period)")
for a in ax[:3]:
    a.axvspan(2400, 2580, color="#d62728", alpha=.12); a.set_xlabel("time window")
plt.tight_layout(); plt.savefig("outputs/figures/preprocessing_steps.png", dpi=150); plt.close()

# ---------- Figure 2: training evidence
ck = torch.load("outputs/models.pt"); sc = pickle.load(open("outputs/scalers.pkl", "rb"))
lstm = LSTMAutoencoder(); lstm.load_state_dict(ck["lstm"]); lstm.eval()
test = res[res.t >= TRAIN_END]
fig, ax = plt.subplots(2, 3, figsize=(16, 8.4))
ax[0, 0].plot(range(1, 26), log["lstm"], marker="o", ms=3, color="#8b5cf6"); ax[0, 0].set(title="LSTM autoencoder: training loss", xlabel="epoch", ylabel="MSE")
ax[0, 1].plot(range(1, 31), log["gnn"], marker="o", ms=3, color="#0891b2"); ax[0, 1].set(title="GNN (GraphSAGE): training loss", xlabel="epoch", ylabel="weighted cross-entropy", yscale="log")
n, a = test[test.label == 0], test[test.label == 1]
ax[0, 2].hist(n.lstm_risk, bins=40, alpha=.7, label="normal windows", color="#2e7d32"); ax[0, 2].hist(a.lstm_risk, bins=40, alpha=.7, label="attack windows", color="#c62828")
ax[0, 2].set(title="LSTM temporal risk on the test period", xlabel="temporal risk (0-100)"); ax[0, 2].legend()
for name, col, c in [("lstm_risk", "LSTM only", "#8b5cf6"), ("gnn_risk", "GNN only", "#0891b2"), ("risk", "LSTM + GNN fused", "#f59e0b")]:
    f, t_, _ = roc_curve(test.label, test[name]); ax[1, 0].plot(f, t_, label=f"{col} (AUC {roc_auc_score(test.label, test[name]):.3f})", color=c)
ax[1, 0].plot([0, 1], [0, 1], "--", color="#999"); ax[1, 0].set(title="ROC curves (test period)", xlabel="false positive rate", ylabel="true positive rate"); ax[1, 0].legend(loc="lower right")
cm = confusion_matrix(test.label, test.risk > 50); ax[1, 1].imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm): ax[1, 1].text(j, i, v, ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
ax[1, 1].set(title="Fused model: confusion matrix (risk > 50)", xticks=[0, 1], yticks=[0, 1], xticklabels=["pred normal", "pred attack"], yticklabels=["true normal", "true attack"])
ax[1, 2].axis("off")
ax[1, 2].text(0, 1, f"Training set-up\n\nSplit: windows < {TRAIN_END} train, later windows test\n\nLSTM autoencoder\n- {log['lstm_sequences']:,} normal sequences only\n- Adam, lr 1e-3, batch 128, 25 epochs\n- hidden size 32, MSE loss\n\nGNN (GraphSAGE)\n- {log['gnn_graphs']:,} graph snapshots\n- Adam, lr 5e-3, 30 epochs, 2 layers\n- attack class weight 8\n\nSeeds fixed (0) for repeatability", va="top", fontsize=11, family="DejaVu Sans")
plt.tight_layout(); plt.savefig("outputs/figures/training_evidence.png", dpi=150); plt.close()
print("figures saved"); print("LSTM loss first/last:", round(log["lstm"][0], 4), round(log["lstm"][-1], 4)); print("GNN loss first/last:", round(log["gnn"][0], 4), round(log["gnn"][-1], 6))
print("confusion matrix [[TN,FP],[FN,TP]]:", cm.tolist())
print("LSTM loss per epoch:", [round(x, 3) for x in log["lstm"]]); print("GNN loss per epoch:", [round(x, 4) for x in log["gnn"]])
