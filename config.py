"""Shared settings for the whole project."""
HOSTS = ["h1", "h2", "h3", "h4", "h5"]
NODES = HOSTS + ["server"]
WINDOW_SEC = 10                      # one telemetry window = 10 seconds

FEATURES = ["pkts_sent", "pkts_recv", "bytes_sent", "bytes_recv",
            "conn_freq", "conn_duration", "unique_dst", "new_dst",
            "auth_attempts", "failed_auth"]

SEQ_LEN = 12                         # LSTM looks at the last 12 windows (2 min)
TRAIN_END = 1500                     # windows before this are used for training

# fusion weights: LSTM, GNN, authentication indicator, traffic indicator
W_LSTM, W_GNN, W_AUTH, W_TRAFFIC = 0.35, 0.35, 0.15, 0.15

# risk band -> level  (0 Normal, 1 Monitored, 2 Restricted, 3 Isolated)
LEVEL_NAMES = ["Normal", "Monitored", "Restricted", "Isolated"]
def level_from_risk(r):
    return 0 if r <= 30 else 1 if r <= 60 else 2 if r <= 80 else 3
