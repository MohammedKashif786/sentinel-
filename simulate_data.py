"""Creates data/telemetry.csv and data/edges.csv without needing Mininet.
Same columns as the Mininet collector, so the rest of the pipeline is identical.

Three attack episodes are injected (host, start window, ramp length, hold length):
  h2 -> training period, h3 -> testing, h4 -> main demo.
"""
import numpy as np, pandas as pd
from config import HOSTS, WINDOW_SEC

rng = np.random.default_rng(42)
T = 3000
EPISODES = [("h2", 800, 100, 60), ("h3", 1800, 100, 60), ("h4", 2400, 100, 80)]
DECAY = 40

# every host has its own normal "personality"
BASE = {  # conn/min, avg bytes per packet, duration(s), usual peers
    "h1": dict(conn=22, size=620, dur=4.0, peers=["h3"]),
    "h2": dict(conn=30, size=540, dur=3.2, peers=[]),
    "h3": dict(conn=15, size=800, dur=6.0, peers=["h1"]),
    "h4": dict(conn=25, size=700, dur=3.8, peers=["h5"]),
    "h5": dict(conn=18, size=500, dur=5.0, peers=["h4"]),
}

def attack_strength(host, t):
    for h, s, ramp, hold in EPISODES:
        if h != host:
            continue
        if s <= t < s + ramp:                return (t - s) / ramp
        if s + ramp <= t < s + ramp + hold:  return 1.0
        if s + ramp + hold <= t < s + ramp + hold + DECAY:
            return 1.0 - (t - s - ramp - hold) / DECAY
    return 0.0

rows, edges = [], []
for t in range(T):
    for h in HOSTS:
        b = BASE[h]; p = attack_strength(h, t)
        daily = 1 + 0.15 * np.sin(2 * np.pi * t / 360)
        conn = b["conn"] * daily * rng.lognormal(0, 0.10) * (1 + 7 * p)
        pk_s = conn * rng.uniform(35, 45) * (1 + 0.6 * p)
        pk_r = pk_s * rng.uniform(0.85, 1.0)
        by_s = pk_s * b["size"] * rng.lognormal(0, 0.05) * (1 + 0.8 * p)
        by_r = pk_r * b["size"] * 1.2 * rng.lognormal(0, 0.05)
        dur = max(0.2, rng.normal(b["dur"], 0.1 * b["dur"]) * (1 - 0.6 * p))
        uniq = 1 + len(b["peers"]) + rng.poisson(0.3) + int(round(8 * p))
        new = rng.poisson(0.05) + (rng.poisson(10 * p) if p > 0.25 else 0)
        auth = rng.poisson(1.5) + (rng.poisson(14 * max(p - 0.35, 0)) if p > 0.35 else 0)
        fail = rng.poisson(0.1) + (rng.poisson(12 * max(p - 0.35, 0)) if p > 0.35 else 0)
        rows.append(dict(t=t, host=h, pkts_sent=pk_s, pkts_recv=pk_r, bytes_sent=by_s,
                         bytes_recv=by_r, conn_freq=conn, conn_duration=dur,
                         unique_dst=uniq, new_dst=new, auth_attempts=auth + fail,
                         failed_auth=fail, label=int(p > 0.2)))
        # communication relationships (who talks to whom)
        edges.append((t, h, "server", max(1, int(conn * WINDOW_SEC / 60 * 0.8))))
        for q in b["peers"]:
            edges.append((t, h, q, max(1, int(rng.poisson(3)))))
        if p > 0.2:                                  # attacker contacts many devices
            for q in HOSTS:
                if q != h:
                    edges.append((t, h, q, max(1, int(rng.poisson(6 * p)) + 1)))

pd.DataFrame(rows).to_csv("data/telemetry.csv", index=False)
pd.DataFrame(edges, columns=["t", "src", "dst", "flows"]).to_csv("data/edges.csv", index=False)
print("saved data/telemetry.csv:", len(rows), "rows | data/edges.csv:", len(edges), "rows")
