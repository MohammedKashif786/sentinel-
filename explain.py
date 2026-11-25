"""Explainability: SHAP on a surrogate model that mimics the fused risk score."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from config import FEATURES

READABLE = {"pkts_sent": "Rapid increase in outbound packets", "pkts_recv": "Unusual inbound packets",
            "bytes_sent": "Rapid increase in outbound traffic", "bytes_recv": "Unusual inbound traffic",
            "conn_freq": "Connection rate far above normal", "conn_duration": "Unusually short connections",
            "unique_dst": "Unusual communication relationships", "new_dst": "Many new destinations",
            "auth_attempts": "Abnormal authentication activity", "failed_auth": "Repeated authentication failures"}

class Explainer:
    def __init__(self, z, risk):
        ok = ~np.isnan(risk)
        self.rf = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=0, n_jobs=-1)
        self.rf.fit(z[ok], risk[ok])
        try:
            import shap; self.sh = shap.TreeExplainer(self.rf)
        except Exception:
            self.sh = None
    def reasons(self, z_row, k=4):
        if self.sh is not None:
            contrib = np.asarray(self.sh.shap_values(z_row.reshape(1, -1))).reshape(-1)
        else:
            contrib = z_row
        top = np.argsort(-contrib)[:k]
        return [READABLE[FEATURES[i]] for i in top if contrib[i] > 0]
