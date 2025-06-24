import json
from typing import List, Dict, Set

def load_knob_data(json_path: str):
    """
    Load the JSON config, collect unique knob types and volumes.
    Returns:
        unique_knob_types (set), unique_volumes (set)
    """
    with open(json_path, 'r') as f:
        dataset: List[Dict] = json.load(f)
    knob_types: Set[str] = set()
    volumes: Set[float] = set()
    for entry in dataset:
        knob_types.add(entry.get('knob_type'))
        volumes.add(entry.get('volume'))
    print(f"Unique knob types: {sorted(knob_types)}")
    print(f"Unique volumes: {sorted(volumes)}")
    return knob_types, volumes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load knob data JSON and print unique knob types and volumes.")
    parser.add_argument("json_path", type=str, help="Path to knob dataset JSON config.")
    args = parser.parse_args()
    load_knob_data(args.json_path) 