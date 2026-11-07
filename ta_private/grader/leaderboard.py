"""Collect runs/*/report.json into leaderboard.csv (macro_f1 desc, ties by mcc; failed at bottom)."""
import json
from pathlib import Path
import pandas as pd

TA = Path(__file__).resolve().parents[1]
rows = []
for d in sorted((TA / "runs").iterdir()):
    if not d.is_dir():
        continue
    rp = d / "report.json"
    r = json.loads(rp.read_text()) if rp.exists() else {"status": "failed", "error": "no report.json"}
    rows.append({"student_id": d.name, "status": r.get("status"),
                 "macro_f1": r.get("macro_f1"), "mcc": r.get("mcc"),
                 "balanced_accuracy": r.get("balanced_accuracy"),
                 "n_scored": r.get("n_scored"), "error": r.get("error", "")})
df = pd.DataFrame(rows, columns=["student_id", "status", "macro_f1", "mcc",
                                 "balanced_accuracy", "n_scored", "error"])
ok = df[df.status == "ok"].sort_values(["macro_f1", "mcc"], ascending=False)
bad = df[df.status != "ok"]
df = pd.concat([ok, bad], ignore_index=True)
df.insert(0, "rank", [i + 1 if s == "ok" else None for i, s in enumerate(df.status)])
df["rank"] = df["rank"].astype("Int64")
df.to_csv(TA / "leaderboard.csv", index=False)
print(df.to_string(index=False))
