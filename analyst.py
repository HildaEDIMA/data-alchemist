"""Muse Spark as an autonomous data analyst: the 'AI alone' baseline, and the input of Audit mode.

The model writes pandas code, we run it and send the printed output back, for a few turns.
It finishes with a JSON list of its discoveries. We add NO hint about p-hacking by default:
this imitates a user who just says "analyze my data".

Safety note: the code written by the model runs on your machine. It runs in a separate process,
with a time limit, but read the first runs. Only run it on data you do not mind exposing.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

MODEL = "muse-spark-1.3"
BASE_URL = "https://api.meta.ai/v1"
MAX_TURNS = 8

SYSTEM = (
    "You are an autonomous data analyst. A pandas DataFrame named `df` is loaded.\n"
    "Explore it to find interesting relationships. Write Python (pandas, numpy, scipy) inside "
    "```python blocks. Each block runs in a FRESH process where `df` is already loaded, and its printed "
    "output is returned to you. You may run several analyses.\n"
    "When you are done, reply with ONLY a JSON list of your findings, each shaped like "
    '{"claim": "...", "variables": ["col_a", "col_b"], "asserts_effect": true, "evidence": "..."}. '
    "asserts_effect is true if the claim says a relationship or difference exists, and false if it says there is none."
)
CAREFUL = (
    "\nBe careful: many comparisons can produce chance findings. Only report discoveries you would trust "
    "after accounting for how many things you tried."
)

_client = None


def ask(messages):
    """One call to Muse Spark through the Meta Model API (OpenAI-compatible)."""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"])
    r = _client.chat.completions.create(model=MODEL, messages=messages)
    return r.choices[0].message.content or ""


def run_code(code, csv_path, timeout=90):
    prelude = f"import pandas as pd, numpy as np\nfrom scipy import stats\ndf = pd.read_csv({csv_path!r})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prelude + code)
        path = f.name
    try:
        out = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        text = out.stdout + out.stderr
    except subprocess.TimeoutExpired:
        text = "[timeout: the code took too long]"
    finally:
        os.unlink(path)
    return text[:4000]


def parse_claims(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        claims = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return [c for c in claims if isinstance(c, dict) and c.get("variables")] if isinstance(claims, list) else None


def analyst(csv_path, careful=False, max_turns=MAX_TURNS):
    """Returns (claims, transcript)."""
    import pandas as pd
    cols = list(pd.read_csv(csv_path, nrows=5).columns)
    msgs = [
        {"role": "system", "content": SYSTEM + (CAREFUL if careful else "")},
        {"role": "user", "content": f"Analyze this dataset and tell me the interesting insights you find.\nColumns: {cols}"},
    ]
    for _ in range(max_turns):
        reply = ask(msgs)
        msgs.append({"role": "assistant", "content": reply})
        blocks = re.findall(r"```python\n(.*?)```", reply, re.S)
        if blocks:
            outs = "\n\n".join(f"[output of block {i + 1}]\n{run_code(b, csv_path)}" for i, b in enumerate(blocks))
            msgs.append({"role": "user", "content": outs})
            continue
        claims = parse_claims(reply)
        if claims is not None:
            return claims, msgs
        msgs.append({"role": "user", "content": "Reply with the JSON list of your discoveries only."})
    msgs.append({"role": "user", "content": "Time is up. Reply NOW with ONLY the JSON list of your discoveries."})
    reply = ask(msgs)
    msgs.append({"role": "assistant", "content": reply})
    return parse_claims(reply) or [], msgs