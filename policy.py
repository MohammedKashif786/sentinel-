"""Natural-language policy engine.
Administrator sentence -> LLM (Ollama) -> JSON -> Pydantic validation -> admin confirmation.
If no LLM is running, a simple rule-based parser is used so the demo always works.
The LLM never touches the network: it only produces a validated rule."""
import json, re
from typing import Literal, Optional
from pydantic import BaseModel, Field

class Rule(BaseModel):
    metric: Literal["risk", "new_destinations", "failed_auth", "conn_freq"]
    operator: Literal[">", ">=", "<", "<="] = ">"
    value: float = Field(ge=0, le=100000)
    window_minutes: Optional[float] = Field(default=None, gt=0, le=60)
    action: Literal["MONITOR", "RESTRICT", "ISOLATE", "HIGH_RISK"]
    text: str = ""

PROMPT = """Convert the security policy into JSON with exactly these keys:
metric (risk | new_destinations | failed_auth | conn_freq), operator (> >= < <=),
value (number), window_minutes (number or null), action (MONITOR | RESTRICT | ISOLATE | HIGH_RISK).
Return ONLY the JSON.
Policy: """

WORDS = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"ten":10,"fifteen":15,"twenty":20,"thirty":30}

def _num(s):
    s = s.lower()
    return float(WORDS[s]) if s in WORDS else float(s)

def _rule_based(text):
    t = text.lower()
    action = ("ISOLATE" if "isolat" in t else "RESTRICT" if "restrict" in t or "block" in t
              else "HIGH_RISK" if "high risk" in t or "high-risk" in t else "MONITOR")
    metric = ("new_destinations" if "destination" in t else "failed_auth" if "fail" in t
              else "conn_freq" if "connection" in t else "risk")
    n = r"(\d+(?:\.\d+)?|" + "|".join(WORDS) + ")"
    v = re.search(r"(?:more than|exceeds?|above|over|greater than|>)\s*" + n, t)
    w = re.search(r"(?:within|in the last|in|over)\s*(?:the last\s*)?" + n + r"\s*minutes?", t)
    if not v: raise ValueError("Could not find a threshold value in the policy.")
    return dict(metric=metric, operator=">", value=_num(v.group(1)),
                window_minutes=_num(w.group(1)) if w else None, action=action)

def parse_policy(text):
    """Returns (Rule, source) where source says whether the LLM or the fallback parser was used."""
    try:
        import ollama
        r = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": PROMPT + text}],
                        format="json", options={"temperature": 0})
        data, src = json.loads(r["message"]["content"]), "LLM (Ollama)"
    except Exception:
        data, src = _rule_based(text), "rule-based fallback"
    data["text"] = text
    return Rule(**data), src          # raises a validation error if the output is unsafe/invalid

def rule_level(rule, host_hist, risk_now):
    """Return the response level (0-3) forced by this rule for one host, or None.
    host_hist: DataFrame of that host's rows up to the current window (needs new_dst, failed_auth...)."""
    n = int(rule.window_minutes * 60 / 10) if rule.window_minutes else 1
    recent = host_hist.tail(n)
    val = {"risk": risk_now, "new_destinations": recent.new_dst.sum(),
           "failed_auth": recent.failed_auth.sum(), "conn_freq": recent.conn_freq.mean()}[rule.metric]
    ok = {">": val > rule.value, ">=": val >= rule.value,
          "<": val < rule.value, "<=": val <= rule.value}[rule.operator]
    if not ok: return None
    return {"MONITOR": 1, "RESTRICT": 2, "ISOLATE": 3, "HIGH_RISK": 2}[rule.action]

if __name__ == "__main__":
    for s in ["Isolate any endpoint whose risk exceeds 90.",
              "If a device communicates with more than 15 new external destinations within five minutes, classify it as high risk."]:
        r, src = parse_policy(s); print(src, "->", r.model_dump_json())
