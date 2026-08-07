"""
Split a dataset root's data.json into one data_XXX.json per training capture.

Each output file trains on exactly one train capture and keeps a single
validation file, so we can check whether each capture is individually
trainable.
"""

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write one test/trainability data_XXX.json file per training capture "
            "for the provided dataset root."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Dataset root containing data.json and a test/ directory.",
    )
    return parser.parse_args()


def _abspath(root: Path, relative_path: str) -> str:
    return str((root / relative_path).resolve())


def main():
    args = _parse_args()
    root = args.root.expanduser().resolve()
    test_dir = root / "test"
    data_path = root / "data.json"
    test_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path) as fp:
        data = json.load(fp)

    validation = [dict(data["validation"][0])]
    validation[0]["x_path"] = _abspath(root, validation[0]["x_path"])
    validation[0]["y_path"] = _abspath(root, validation[0]["y_path"])

    for i, train_entry in enumerate(data["train"]):
        entry = dict(train_entry)
        entry["x_path"] = _abspath(root, entry["x_path"])
        entry["y_path"] = _abspath(root, entry["y_path"])
        out = {
            "type": data["type"],
            "common": data.get("common", {}),
            "train": [entry],
            "validation": validation,
        }
        path = test_dir / f"data_{i:03d}.json"
        with open(path, "w") as fp:
            json.dump(out, fp, indent=4)
        print(f"wrote {path.name}  ({Path(entry['y_path']).name})")


if __name__ == "__main__":
    main()
