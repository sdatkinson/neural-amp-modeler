import os
import json
import numpy as np
import wavio
from typing import List, Dict

def create_dummy_wav(path: str, length: int = 16000, value: float = 0.1, rate: int = 48000, sampwidth: int = 3):
    data = np.full((length,), value, dtype=np.float32)
    # wavio expects int, so scale and cast
    scaled = (np.clip(data, -1.0, 1.0) * (2 ** (8 * sampwidth - 1))).astype(np.int32)
    wavio.write(path, scaled, rate, sampwidth=sampwidth)

def generate_knob_data(
    out_dir: str,
    num_pairs: int = 5,
    wav_length: int = 16000,
    rate: int = 48000,
    sampwidth: int = 3,
    json_name: str = "knob_dataset.json"
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    knob_types = ["gain", "bass", "mid", "treble", "presence"]
    dataset: List[Dict] = []
    for i in range(num_pairs):
        input_path = os.path.join(out_dir, f"input_{i+1}.wav")
        output_path = os.path.join(out_dir, f"output_{i+1}.wav")
        knob_type = knob_types[i % len(knob_types)]
        knob_level = round(np.random.uniform(0.0, 1.0), 2)
        volume = round(np.random.uniform(0.2, 1.0), 2)
        create_dummy_wav(input_path, length=wav_length, value=0.1 + 0.1 * i, rate=rate, sampwidth=sampwidth)
        create_dummy_wav(output_path, length=wav_length, value=0.2 + 0.1 * i, rate=rate, sampwidth=sampwidth)
        dataset.append({
            "input_path": input_path,
            "output_path": output_path,
            "knob_type": knob_type,
            "knob_level": knob_level,
            "volume": volume
        })
    json_path = os.path.join(out_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=2)
    return json_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate dummy knob data and JSON config.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for WAVs and JSON config.")
    parser.add_argument("--num_pairs", type=int, default=5, help="Number of input/output pairs to generate.")
    args = parser.parse_args()
    json_path = generate_knob_data(args.out_dir, num_pairs=args.num_pairs)
    print(f"Generated dataset JSON at {json_path}") 