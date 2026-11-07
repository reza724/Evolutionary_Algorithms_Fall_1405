"""Local replay only: executes trusted student code, not a security sandbox.

The split is chronological. Train separately on rows before train_end_exclusive;
the purge gap is excluded from training and validation scoring. Earlier feature
rows may be used as causal context, without any labels.
"""
import argparse
import importlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

if __package__:
    from .metrics import evaluate
else:
    from metrics import evaluate


def validate_manifest(package):
    manifest = json.loads((package / "manifest.json").read_text())
    schema = json.loads((Path(__file__).resolve().parents[1] /
                         "schemas/manifest.schema.json").read_text())
    try:
        import jsonschema
    except ImportError:
        if not isinstance(manifest, dict) or set(manifest) != set(schema["required"]):
            raise ValueError("Invalid manifest keys")
        for key, rule in schema["properties"].items():
            value = manifest[key]
            if "const" in rule and value != rule["const"]:
                raise ValueError("Invalid manifest value: " + key)
            if rule.get("type") == "string" and not isinstance(value, str):
                raise ValueError("Expected string: " + key)
            if "pattern" in rule and not re.search(rule["pattern"], value):
                raise ValueError("Invalid manifest format: " + key)
    else:
        jsonschema.Draft202012Validator(schema).validate(manifest)
    for key in ("artifacts_dir", "requirements_file"):
        resolved = (package / manifest[key]).resolve()
        if not resolved.is_relative_to(package):
            raise ValueError("Package path escapes root")
        if key == "artifacts_dir" and not resolved.is_dir():
            raise ValueError("Missing artifacts directory")
        if key == "requirements_file" and not resolved.is_file():
            raise ValueError("Missing requirements file")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--purge", type=int, default=600)
    parser.add_argument("--lookback", type=int, default=288)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.val_frac < 1 or args.purge < 0 or args.lookback < 1:
        parser.error("Invalid split, purge, or lookback")
    package = args.package.resolve()
    manifest = validate_manifest(package)
    data = pd.read_csv(args.data)
    if "label" not in data:
        raise ValueError("Missing target column label")
    start = int(len(data) * (1 - args.val_frac))
    train_end = start - args.purge
    if train_end <= 0 or start >= len(data):
        raise ValueError("Insufficient rows for requested split and purge")
    features = data.drop(columns=["label"])
    # Supplied order is the chronological contract; never reorder features.
    sys.path.insert(0, str(package))
    module_name, class_name = manifest["entrypoint"].split(":")
    module = importlib.import_module(module_name)
    if not Path(module.__file__).resolve().is_relative_to(package):
        raise ValueError("Entrypoint must be inside package")
    model = getattr(module, class_name)(str(package / manifest["artifacts_dir"]))
    model.reset(args.seed)
    predictions = []
    for i in range(start, len(data)):
        window = features.iloc[max(0, i - args.lookback + 1):i + 1].copy(deep=True)
        now = features.iloc[i]["time"] if "time" in features else i
        prediction = model.update(window, now)
        if isinstance(prediction, bool) or not isinstance(prediction, int) or prediction not in (-1, 0, 1):
            raise ValueError(f"Invalid prediction at row {i}: {prediction!r}")
        predictions.append(prediction)
    result = evaluate(data["label"].iloc[start:].to_numpy(), predictions)
    result["split"] = {"train_end_exclusive": train_end,
                       "validation_start": start, "purge": args.purge,
                       "lookback": args.lookback, "n_predictions": len(predictions)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
