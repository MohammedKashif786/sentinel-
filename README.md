# ML-Driven Adaptive Endpoint Security (LSTM + GNN + NLP policies)

## Path A - runs on Windows directly (no Mininet needed)
Uses a Python simulator that produces the same telemetry format as the Mininet lab.

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python simulate_data.py        # 1. creates data/telemetry.csv + data/edges.csv
    python train.py                # 2. trains LSTM autoencoder + GraphSAGE GNN
    python run_pipeline.py         # 3. risk scores, fusion, policies, response, plots, metrics
    streamlit run dashboard.py     # 4. live dashboard

If `pip install torch_geometric` complains, install torch first, then torch_geometric.
The LLM is optional: with Ollama running (`ollama pull llama3.2`) policies are parsed by the LLM,
otherwise a rule-based parser is used automatically.

## Path B - real Mininet network (Ubuntu VM or WSL2)
    sudo apt install mininet openvswitch-switch curl nmap iperf3
    sudo python3 mininet_lab/topology.py     # builds topology, generates traffic, writes data/*.csv
then run steps 2-4 above on the new data (use the Python venv, not sudo).
`mininet_lab/response.py` shows how Restrict / Isolate / Restore become Open vSwitch flow rules.

## Output files (outputs/)
- risk_timeline.png     risk of all hosts + adaptive response of h4
- model_scores_h4.png   LSTM, GNN and fused risk for h4
- results.csv           every host and window with all scores and status
- pipeline_output.txt   console report: metrics, policies, response log
