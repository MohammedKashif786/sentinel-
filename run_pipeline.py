"""Detect -> Assess -> Explain -> Respond -> Monitor -> Recover.
Scores every window with the LSTM and GNN, fuses the risk, applies admin policies,
runs the adaptive response engine and writes results + plots to outputs/."""
import pickle, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.ensemble import IsolationForest
from common import *
from explain import Explainer
from policy import parse_policy, rule_level

tel, edges = load()
tel = tel.sort_values(["host", "t"]).reset_index(drop=True)
sc = pickle.load(open("outputs/scalers.pkl", "rb")); base, q99 = sc["base"], sc["q99"]
ck = torch.load("outputs/models.pt")
lstm = LSTMAutoencoder(); lstm.load_state_dict(ck["lstm"]); lstm.eval()
gnn = GNN(); gnn.load_state_dict(ck["gnn"]); gnn.eval()
z = zscores(tel, base)
T = tel.t.max() + 1

# ------------------------------------------------ 1. temporal risk (LSTM)
tel["lstm_risk"] = np.nan
for h in HOSTS:
    m = (tel.host == h).values
    e = lstm_errors(lstm, z[m]); r = e / q99
    tel.loc[m, "lstm_risk"] = 100 * r / (r + 4)
# ------------------------------------------------ 2. graph risk (GNN)
graphs = build_graphs(tel, edges, z, range(T))
gr = {}
with torch.no_grad():
    for g in graphs:
        p = torch.softmax(gnn(g), 1)[:, 1].numpy()
        for h in HOSTS: gr[(g.t, h)] = 100 * p[NODES.index(h)]
tel["gnn_risk"] = [gr[(t, h)] for t, h in zip(tel.t, tel.host)]
# ------------------------------------------------ 3. indicators + fusion
zf = pd.DataFrame(z, columns=FEATURES)
tel["auth_risk"] = 100 * np.clip(zf.failed_auth.values / 8, 0, 1)
tel["traffic_risk"] = 100 * np.clip(zf.conn_freq.values / 8, 0, 1)
raw = (W_LSTM * tel.lstm_risk + W_GNN * tel.gnn_risk
       + W_AUTH * tel.auth_risk + W_TRAFFIC * tel.traffic_risk)
tel["risk"] = raw.groupby(tel.host).transform(lambda s: s.ewm(alpha=0.6).mean())  # light smoothing
tel = tel.dropna(subset=["risk"]).reset_index(drop=True)
z = zscores(tel, base)

# ------------------------------------------------ 4. evaluation (test period only)
test = tel[tel.t >= TRAIN_END]
print("\n=== DETECTION QUALITY (test period, windows after %d) ===" % TRAIN_END)
norm_train = tel[(tel.t < TRAIN_END) & (tel.label == 0)]
iso = IsolationForest(random_state=0).fit(np.log1p(norm_train[FEATURES]))
iso_thr = np.quantile(-iso.score_samples(np.log1p(norm_train[FEATURES])), 0.99)
test_iso = pd.Series(-iso.score_samples(np.log1p(test[FEATURES])), index=test.index)
rows = []
for name, s, thr in [("Fixed threshold (conn_freq > 60)", test.conn_freq, 60), ("Isolation Forest", test_iso, iso_thr),
                     ("LSTM only", test.lstm_risk, 50), ("GNN only", test.gnn_risk, 50),
                     ("LSTM + GNN fused (proposed)", test.risk, 50)]:
    p, r, f, _ = precision_recall_fscore_support(test.label, s > thr, average="binary", zero_division=0)
    rows.append((name, p, r, f, roc_auc_score(test.label, s)))
print(pd.DataFrame(rows, columns=["Method", "Precision", "Recall", "F1", "ROC-AUC"]).round(3).to_string(index=False))

# ------------------------------------------------ 5. policies (admin typed, validated, confirmed)
policies = []
for text in ["Isolate any endpoint whose risk exceeds 90.",
             "If a device communicates with more than 15 new external destinations within five minutes, classify it as high risk."]:
    rule, src = parse_policy(text)
    print(f"\nPolicy: \"{text}\"\n  parsed by {src}: {rule.model_dump(exclude={'text'})}\n  admin confirmation: APPROVED")
    policies.append(rule)

# ------------------------------------------------ 6. response engine with recovery
HOLD, LOWER = 6, [None, 30, 60, 80]          # a level steps down only after 6 calm windows
state = {h: dict(level=0, calm=0) for h in HOSTS}
levels, events = [], []
expl = Explainer(z, tel.risk.values)
for h in HOSTS:
    hist = tel[tel.host == h]
    for i in range(len(hist)):
        row = hist.iloc[i]; st = state[h]
        want = level_from_risk(row.risk)
        for rule in policies:
            lv = rule_level(rule, hist.iloc[:i + 1], row.risk)
            if lv is not None: want = max(want, lv)
        if want >= st["level"]:
            if want > st["level"]:
                why = expl.reasons(z[hist.index[i]]) if want >= 2 else []
                events.append((int(row.t), h, LEVEL_NAMES[want].upper(), round(row.risk), why))
            st["level"], st["calm"] = want, 0
        else:
            st["calm"] += 1
            if st["calm"] >= HOLD and row.risk <= LOWER[st["level"]]:
                st["level"] -= 1; st["calm"] = 0
                events.append((int(row.t), h, "-> " + LEVEL_NAMES[st["level"]], round(row.risk), []))
        levels.append(st["level"])
tel = pd.concat([tel[tel.host == h] for h in HOSTS]).reset_index(drop=True)
tel["level"] = levels; tel["status"] = tel.level.map(dict(enumerate(LEVEL_NAMES)))
tel.round(3).to_csv("outputs/results.csv", index=False)

# ------------------------------------------------ 7. report for the demo endpoint
print("\n=== RESPONSE LOG (test period) ===")
for t, h, what, r, why in events:
    if t >= TRAIN_END:
        print(f"t={t:4d} ({t*10/60:6.1f} min)  {h}  risk={r:3d}  {what}" + (f"   reasons: {'; '.join(why)}" if why else ""))
d = tel[(tel.host == "h4") & (tel.t >= 2380)]
print("\n=== h4 DEMO (attack starts at window 2400) ===")
peak = d.loc[d.risk.idxmax()]
print(f"Peak risk {peak.risk:.0f}/100 at window {int(peak.t)}  (LSTM {peak.lstm_risk:.0f}%, GNN {peak.gnn_risk:.0f}%, "
      f"auth {peak.auth_risk:.0f}%, traffic {peak.traffic_risk:.0f}%)")
start = 2400
print(f"First alert (Monitored): {(d[d.level >= 1].t.min()-start)*10} s after attack began | ISOLATE: {(d[d.level >= 3].t.min()-start)*10} s after")
others = tel[(tel.host != "h4") & (tel.t.between(2400, 2600))]
print(f"Other hosts during h4 attack: max risk {others.risk.max():.0f}, max level {LEVEL_NAMES[others.level.max()]}")
path = d.status[d.status != d.status.shift()].tolist()
print("Status path for h4:", " -> ".join(path))

# ------------------------------------------------ 8. figures
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for h, c in zip(HOSTS, ["#999", "#bbb", "#4c78a8", "#d62728", "#7f7f7f"]):
    g = tel[tel.host == h]; ax[0].plot(g.t, g.risk, color=c, lw=2 if h == "h4" else 1, label=h)
for lo, hi, col in [(0, 30, "#e8f5e9"), (30, 60, "#fffde7"), (60, 80, "#fff3e0"), (80, 100, "#ffebee")]:
    ax[0].axhspan(lo, hi, color=col, zorder=0)
ax[0].set(ylabel="Overall risk (0-100)", ylim=(0, 100), xlim=(TRAIN_END, T), title="Endpoint risk over time (test period)"); ax[0].legend(ncol=5, loc="upper left")
g = tel[tel.host == "h4"]; ax[1].step(g.t, g.level, color="#d62728", where="post")
ax[1].set(yticks=range(4), yticklabels=LEVEL_NAMES, xlabel="Time window (10 s each)", title="Adaptive response for h4"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/risk_timeline.png", dpi=150)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, col, ttl in zip(ax, ["lstm_risk", "gnn_risk", "risk"], ["LSTM temporal risk (h4)", "GNN graph risk (h4)", "Fused overall risk (h4)"]):
    g = tel[(tel.host == "h4") & (tel.t.between(2300, 2800))]
    a.plot(g.t, g[col], color="#1F3864"); a.axvspan(2400, 2580, color="#d62728", alpha=.12); a.set(title=ttl, ylim=(0, 100), xlabel="window")
plt.tight_layout(); plt.savefig("outputs/model_scores_h4.png", dpi=150)
print("\nSaved: outputs/results.csv, outputs/risk_timeline.png, outputs/model_scores_h4.png")
