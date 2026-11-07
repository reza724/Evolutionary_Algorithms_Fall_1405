"""Run one student submission on the private Benchmark 1 test set.

Usage: python grader/run_submission.py submissions/<student_id>.zip [--timeout-min 30]
Writes runs/<student_id>/{predictions.csv, log.txt, status.json}.
The model runs in a separate subprocess. This is NOT a full sandbox.
"""
import argparse, json, os, shutil, stat, subprocess, sys, tempfile, time, traceback, zipfile
from pathlib import Path

TA = Path(__file__).resolve().parents[1]
PROJECT = TA.parent
STUDENT = PROJECT / "student_release"
TRAIN = STUDENT / "data/train_benchmark_1.csv"
TEST_FEATURES = TA / "data/test_benchmark_1_features.csv"
LOOKBACK = 288
MAX_BYTES = 200 * 1024 * 1024

WORKER = r'''
import importlib, sys, time
import pandas as pd
pkg, student, train, test, out, lookback = sys.argv[1:7]
lookback = int(lookback)
sys.path.insert(0, student)
from ea_sdk.replay import validate_manifest
from pathlib import Path
package = Path(pkg).resolve()
manifest = validate_manifest(package)
sys.path.insert(0, str(package))
mod_name, cls_name = manifest["entrypoint"].split(":")
module = importlib.import_module(mod_name)
if not Path(module.__file__).resolve().is_relative_to(package):
    raise ValueError("Entrypoint must be inside package")
model = getattr(module, cls_name)(str(package / manifest["artifacts_dir"]))
model.reset(seed=0)
tr = pd.read_csv(train).drop(columns=["label"]).tail(lookback)
te = pd.read_csv(test)
if list(tr.columns) != list(te.columns):
    raise ValueError("Train/test feature columns differ")
feats = pd.concat([tr, te], ignore_index=True)
n_warm = len(tr)
def check(p, i):
    if isinstance(p, bool) or not isinstance(p, int) or p not in (-1, 0, 1):
        raise ValueError(f"Invalid prediction at row {i}: {p!r}")
for i in range(n_warm):  # warm-up, not scored
    w = feats.iloc[max(0, i - lookback + 1):i + 1].copy(deep=True)
    check(model.update(w, feats.at[i, "time"]), i)
t0 = time.time()
with open(out, "w") as f:
    f.write("time,y_pred\n")
    for i in range(n_warm, len(feats)):
        w = feats.iloc[max(0, i - lookback + 1):i + 1].copy(deep=True)
        now = feats.at[i, "time"]
        p = model.update(w, now)
        check(p, i - n_warm)
        f.write(f"{now},{p}\n")
        if (i - n_warm) % 10000 == 0:
            print(f"row {i - n_warm}/{len(te)} {time.time() - t0:.0f}s", flush=True)
print("done", len(te), "predictions", flush=True)
'''


def safe_extract(zpath, dest):
    if zpath.stat().st_size > MAX_BYTES:
        raise ValueError("ZIP larger than 200MB")
    with zipfile.ZipFile(zpath) as z:
        total = 0
        for info in z.infolist():
            name = info.filename
            if name.startswith("/") or "\\" in name or ".." in Path(name).parts or ":" in name:
                raise ValueError(f"Unsafe path in ZIP: {name}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Symlink in ZIP: {name}")
            total += info.file_size
            if total > MAX_BYTES:
                raise ValueError("Uncompressed size larger than 200MB")
        z.extractall(dest)
    for p in Path(dest).rglob("*"):
        if p.is_symlink():
            raise ValueError(f"Symlink after extraction: {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip", type=Path)
    ap.add_argument("--student-id")
    ap.add_argument("--timeout-min", type=float, default=30)
    a = ap.parse_args()
    sid = a.student_id or a.zip.stem
    run = TA / "runs" / sid
    run.mkdir(parents=True, exist_ok=True)
    (run / "predictions.csv").unlink(missing_ok=True)
    log = open(run / "log.txt", "w")
    status = {"student_id": sid, "zip": str(a.zip), "status": "failed"}
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix=f"grade_{sid}_")
    try:
        safe_extract(a.zip.resolve(), tmp)
        proc = subprocess.run(
            [sys.executable, "-c", WORKER, tmp, str(STUDENT), str(TRAIN),
             str(TEST_FEATURES), str(run / "predictions.csv"), str(LOOKBACK)],
            cwd=tmp, stdout=log, stderr=subprocess.STDOUT,
            timeout=a.timeout_min * 60)
        if proc.returncode != 0:
            raise RuntimeError(f"Model process exited with code {proc.returncode}")
        status["status"] = "ok"
    except subprocess.TimeoutExpired:
        status["error"] = f"Timeout after {a.timeout_min} min"
    except Exception as e:
        status["error"] = f"{type(e).__name__}: {e}"
        log.write(traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        status["seconds"] = round(time.time() - t0, 1)
        if status["status"] != "ok":
            (run / "predictions.csv").unlink(missing_ok=True)
        log.close()
        (run / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    sys.exit(0 if status["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
