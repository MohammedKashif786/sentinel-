"""Streamlit dashboard:  streamlit run dashboard.py   (run after run_pipeline.py)"""
import pandas as pd, networkx as nx, matplotlib.pyplot as plt, streamlit as st
from config import HOSTS, NODES, LEVEL_NAMES
from policy import parse_policy

st.set_page_config(page_title="Adaptive Endpoint Security", layout="wide")
res = pd.read_csv("outputs/results.csv"); edges = pd.read_csv("data/edges.csv")
st.title("ML-Driven Adaptive Endpoint Security")

t = st.slider("Time window (10 s each)", int(res.t.min()), int(res.t.max()), 2500)
now = res[res.t == t].set_index("host")
COL = ["#2e7d32", "#f9a825", "#ef6c00", "#c62828"]

left, right = st.columns([1, 1])
with left:
    st.subheader("Network topology")
    G = nx.Graph(); G.add_nodes_from(NODES)
    e = edges[(edges.t == t) & (edges.flows >= 2)]
    G.add_edges_from(zip(e.src, e.dst))
    pos = {"server": (0, 1), "h1": (-1.5, 0), "h2": (-0.5, 0), "h3": (0.5, 0), "h4": (1.5, 0), "h5": (2.5, 0)}
    colors = [COL[int(now.loc[n, "level"])] if n in HOSTS else "#1F3864" for n in G.nodes]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    nx.draw(G, pos, with_labels=True, node_color=colors, font_color="white", node_size=1300, ax=ax)
    st.pyplot(fig)
    st.caption("Green = normal, yellow = monitored, orange = restricted, red = isolated")
with right:
    st.subheader("Endpoint status")
    st.dataframe(now[["risk", "lstm_risk", "gnn_risk", "auth_risk", "status"]].round(0).astype(
        {"risk": int, "lstm_risk": int, "gnn_risk": int, "auth_risk": int}), use_container_width=True)
    h = st.selectbox("Risk history of", HOSTS, index=3)
    g = res[res.host == h].set_index("t")
    st.line_chart(g[["risk", "lstm_risk", "gnn_risk"]])

st.subheader("Administrator policy (natural language)")
txt = st.text_input("Type a rule", "Isolate any endpoint whose risk exceeds 90.")
if st.button("Convert to rule"):
    try:
        rule, src = parse_policy(txt)
        st.session_state["rule"] = rule
        st.success(f"Parsed by {src}"); st.json(rule.model_dump(exclude={'text'}))
    except Exception as ex:
        st.error(f"Rejected: {ex}")
if "rule" in st.session_state and st.button("Confirm and activate"):
    st.success("Rule validated and approved. In the pipeline it is applied by the policy engine, "
               "never directly by the LLM.")
