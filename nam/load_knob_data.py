import json
from typing import List, Dict

def load_knob_data(json_path: str) -> List[Dict]:
    """
    Load the JSON config and return the dataset (list of dicts).
    Also prints unique knob types and knob levels for user info.
    """
    with open(json_path, 'r') as f:
        dataset: List[Dict] = json.load(f)
    knob_types = {entry.get('knob_type') for entry in dataset}
    knob_levels = {entry.get('knob_level') for entry in dataset}
    print(f"Unique knob types: {sorted(knob_types)}")
    print(f"Unique knob levels: {sorted(knob_levels)}")
    return dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load knob data JSON and print unique knob types and knob levels.")
    parser.add_argument("json_path", type=str, help="Path to knob dataset JSON config.")
    args = parser.parse_args()
    load_knob_data(args.json_path) 