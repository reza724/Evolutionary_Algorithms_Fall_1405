"""Score runs/<student_id>/predictions.csv against the private labels.

Usage: python grader/evaluate.py <student_id>
Writes runs/<student_id>/report.json (marks status failed on any problem).
"""
import json, sys, traceback
from pathlib import Path
import pandas as pd

TA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TA.parent / "student_release"))
from ea_sdk.metrics import evaluate  # same metric code students use

LABELS = TA / "data/test_benchmark_1_labels.csv"


def main():
    sid = sys.argv[1]
    run = TA / "runs" / sid
    report = {"student_id": sid, "status": "failed"}
    try:
        st = json.loads((run / "status.json").read_text())
        if st.get("status") != "ok":
            raise RuntimeError("Run failed: " + st.get("error", "unknown"))
        pred = pd.read_csv(run / "predictions.csv")
        lab = pd.read_csv(LABELS)
        if list(pred.columns) != ["time", "y_pred"]:
            raise ValueError("predictions.csv must have columns time,y_pred")
        if pred["time"].duplicated().any():
            raise ValueError("Duplicate time in predictions")
        if set(pred["time"]) != set(lab["time"]) or len(pred) != len(lab):
            raise ValueError("Prediction keys do not match test keys exactly")
        m = lab.merge(pred, on="time", how="inner", validate="one_to_one")
        m = m[m["label"].notna()]
        res = evaluate(m["label"].astype(int).to_numpy(), m["y_pred"].to_numpy())
        report.update(status="ok", n_predictions=int(len(pred)), **res)
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    run.mkdir(parents=True, exist_ok=True)
    (run / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report.get(k) for k in
                      ("student_id", "status", "macro_f1", "balanced_accuracy", "mcc", "error")}, indent=2))
    sys.exit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
