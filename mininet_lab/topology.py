#!/usr/bin/env python3
"""Mininet lab: 5 endpoints, 2 switches, 1 router, 1 server  +  traffic generator + telemetry collector.
Run on Linux (Ubuntu VM or WSL2):   sudo python3 mininet_lab/topology.py
Writes ../data/telemetry.csv and ../data/edges.csv in the same format as simulate_data.py.
NOTE: written against the standard Mininet API; test it on your machine and adjust timings if needed.

      h1 h2 h3 --s1--[r1]--s2-- h4 h5 server
"""
import os, sys, time, random, subprocess, collections
from mininet.net import Mininet
from mininet.node import Node, OVSSwitch
from mininet.topo import Topo
from mininet.log import setLogLevel

WINDOW = 10                 # seconds per telemetry window
NORMAL_MIN = 12             # minutes of normal traffic first
ATTACKS = [("h2", 3), ("h3", 3), ("h4", 3)]    # (attacker, minutes) - one after another, with 2 min of normal in between
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
SERVER_IP = "10.0.2.100"

class LinuxRouter(Node):
    def config(self, **p):
        super().config(**p); self.cmd("sysctl net.ipv4.ip_forward=1")
    def terminate(self):
        self.cmd("sysctl net.ipv4.ip_forward=0"); super().terminate()

class Lab(Topo):
    def build(self):
        r1 = self.addNode("r1", cls=LinuxRouter, ip="10.0.1.1/24")
        s1, s2 = self.addSwitch("s1"), self.addSwitch("s2")
        self.addLink(s1, r1, intfName2="r1-eth1", params2={"ip": "10.0.1.1/24"})
        self.addLink(s2, r1, intfName2="r1-eth2", params2={"ip": "10.0.2.1/24"})
        for i, (sw, sub) in enumerate([(s1, 1), (s1, 1), (s1, 1), (s2, 2), (s2, 2)], start=1):
            self.addHost(f"h{i}", ip=f"10.0.{sub}.{10+i}/24", defaultRoute=f"via 10.0.{sub}.1")
            self.addLink(f"h{i}", sw)
        self.addHost("server", ip=f"{SERVER_IP}/24", defaultRoute="via 10.0.2.1")
        self.addLink("server", s2)

RATE = {"h1": 22, "h2": 30, "h3": 15, "h4": 25, "h5": 18}          # normal requests per minute

def start_normal(net):
    for h, r in RATE.items():           # each host has its own personality
        net.get(h).cmd(f"(while true; do curl -s -m 2 -o /dev/null http://{SERVER_IP}/; "
                       f"sleep $(python3 -c 'import random;print(random.expovariate({r}/60))'); done) &")
    net.get("h1").cmd("(while true; do ping -c1 -W1 10.0.1.13 >/dev/null; sleep 4; done) &")   # h1 <-> h3
    net.get("h4").cmd("(while true; do ping -c1 -W1 10.0.2.15 >/dev/null; sleep 4; done) &")   # h4 <-> h5

def attack(net, host, seconds):
    h = net.get(host); end = time.time() + seconds
    h.cmd("pkill -f 'curl -s -m 2' ; true")
    while time.time() < end:            # fast connections + scanning other devices + failed logins
        frac = 1 - max(0, end - time.time()) / seconds
        h.cmd(f"for i in $(seq 1 {int(5 + 60*frac)}); do curl -s -m 1 -o /dev/null http://{SERVER_IP}/ & done; wait")
        if frac > .25: h.cmd("nmap -sn -T4 10.0.1.0/24 10.0.2.0/24 >/dev/null 2>&1")
        if frac > .35: h.cmd(f"for i in $(seq 1 {int(20*frac)}); do curl -s -m 1 -X POST http://{SERVER_IP}/login -d x=1 -o /dev/null & done; wait")
        time.sleep(1)
    h.cmd("pkill nmap; true"); start_normal(net)       # back to normal afterwards

def read_dev(h):
    """packets / bytes sent and received on the host's interface, from /proc/net/dev"""
    for line in h.cmd("cat /proc/net/dev").splitlines():
        if line.strip().startswith(f"{h.name}-eth0"):
            f = line.split(":")[1].split(); return int(f[1]), int(f[0]), int(f[9]), int(f[8])  # rx_p, rx_b, tx_p, tx_b
    return 0, 0, 0, 0

def connections(h):
    out = h.cmd("ss -tn").splitlines()[1:]
    return [l.split()[4].rsplit(":", 1)[0] for l in out if l.strip()]

def collect(net, stop_time, labels):
    names = [f"h{i}" for i in range(1, 6)]; ip2h = {net.get(n).IP(): n for n in names}
    prev = {n: read_dev(net.get(n)) for n in names}; seen = {n: set() for n in names}
    log_pos = 0; rows, edges = [], []; t = 0
    while time.time() < stop_time:
        time.sleep(WINDOW)
        failed = collections.Counter(); logtxt = open("/tmp/server_log.txt").read().splitlines()[log_pos:]; log_pos += len(logtxt)
        for l in logtxt:
            ts, ip, path, code = l.split()
            if code == "401": failed[ip2h.get(ip)] += 1
        for n in names:
            h = net.get(n); cur = read_dev(h); d = [c - p for c, p in zip(cur, prev[n])]; prev[n] = cur
            dst = connections(h); uniq = set(dst); new = uniq - seen[n]; seen[n] |= uniq
            nconn = len(dst) + 0.0
            rows.append(dict(t=t, host=n, pkts_sent=d[2], pkts_recv=d[0], bytes_sent=d[3], bytes_recv=d[1],
                             conn_freq=nconn * 60 / WINDOW, conn_duration=WINDOW / max(nconn, 1),  # rough estimate
                             unique_dst=len(uniq), new_dst=len(new) if t > 3 else 0,
                             auth_attempts=failed[n] + 1, failed_auth=failed[n], label=labels(t)))
            for ip, c in collections.Counter(dst).items():
                edges.append((t, n, "server" if ip == SERVER_IP else ip2h.get(ip, ip), c))
        t += 1
    return rows, edges

if __name__ == "__main__":
    import pandas as pd, threading
    setLogLevel("info")
    net = Mininet(topo=Lab(), switch=OVSSwitch, controller=None, autoSetMacs=True); net.start()
    for sw in ("s1", "s2"): net.get(sw).cmd(f"ovs-vsctl set-fail-mode {sw} standalone")
    print("Connectivity test:"); net.pingAll()
    net.get("server").cmd(f"python3 {HERE}/server_app.py /tmp/server_log.txt &"); open("/tmp/server_log.txt", "w").close()
    time.sleep(1); start_normal(net)

    timeline, t0 = [], time.time()                     # (attacker, start_sec, end_sec) for ground-truth labels
    cur = NORMAL_MIN * 60
    for who, mins in ATTACKS:
        timeline.append((who, cur, cur + mins * 60)); cur += (mins + 2) * 60
    total = cur
    def label_fn(host_idx):                             # label by time window (host handled below)
        return 0
    def run_attacks():
        for who, s, e in timeline:
            time.sleep(max(0, t0 + s - time.time())); attack(net, who, e - s)
    threading.Thread(target=run_attacks, daemon=True).start()
    rows, edges = collect(net, t0 + total, label_fn)
    # ground truth: windows where that host was the attacker
    for r in rows:
        for who, s, e in timeline:
            if r["host"] == who and s <= r["t"] * WINDOW < e: r["label"] = 1
    os.makedirs(DATA, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "telemetry.csv"), index=False)
    pd.DataFrame(edges, columns=["t", "src", "dst", "flows"]).to_csv(os.path.join(DATA, "edges.csv"), index=False)
    print("Saved telemetry from the live Mininet network."); net.stop()
