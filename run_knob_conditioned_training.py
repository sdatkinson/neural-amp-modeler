"""
run_knob_conditioned_training.py

This script runs an end-to-end test of the knob-conditioned training pipeline using the example configs and knob JSON files.

Instructions:
1. Ensure you have installed all dependencies (see requirements.txt or environments/).
2. Make sure you have the following files in place:
   - nam_full_configs/knobs/single_knob.json (or two_knobs.json)
   - nam_full_configs/models/wavenet.json (or your custom model config)
   - nam_full_configs/learning/default.json (or your custom learning config)
   - data/knob_dataset.json (or knob_train.json and knob_val.json for two_knobs)
   - The referenced WAV files in the data/ directory
3. Run this script from the project root:

   python run_knob_conditioned_training.py

You can change the config paths below to use your own files.
"""

import os
from pathlib import Path
from nam.train.full import main
import logging
from nam.load_knob_data import load_knob_data
from nam.train.core import get_wavenet_config, Architecture

logging.getLogger('matplotlib').setLevel(logging.WARNING)

logging.basicConfig(level=logging.DEBUG)

# === CONFIGURATION ===
# Change these paths as needed
DATA_CONFIG = "nam_full_configs/knobs/single_knob.json"  # or "two_knobs.json"
MODEL_CONFIG = "nam_full_configs/models/wavenet.json"
LEARNING_CONFIG = "nam_full_configs/learning/default.json"
OUTDIR = "output_knob_conditioned_test"

# === SCRIPT ===
if __name__ == "__main__":
    import json

    # Create output directory if it doesn't exist
    outdir = Path(OUTDIR)
    outdir.mkdir(exist_ok=True)

    # Load configs
    with open(DATA_CONFIG, "r") as f:
        data_config = json.load(f)
    with open(MODEL_CONFIG, "r") as f:
        model_config = json.load(f)
    with open(LEARNING_CONFIG, "r") as f:
        learning_config = json.load(f)

    # Path to your knob dataset JSON
    json_path = 'data/knob_dataset.json'

    # Load the knob data to determine the number of conditioning channels
    entries = load_knob_data(json_path)
    knob_types = sorted({entry['knob_type'] for entry in entries})  # Sort for consistency
    condition_size = len(knob_types) + 1  # +1 for knob value

    # Set architecture to default (STANDARD)
    architecture = Architecture.STANDARD

    # Get the WaveNet config
    wavenet_config = get_wavenet_config(architecture)

    # Update condition_size for all layers in the config
    if 'layers' in wavenet_config:
        for layer in wavenet_config['layers']:
            if 'condition_size' in layer:
                layer['condition_size'] = condition_size

    # Update condition_size for all layers in the loaded model config
    if 'net' in model_config and 'config' in model_config['net']:
        # Add knob_types to the model config
        model_config['net']['config']['knob_types'] = knob_types
        
        layers_key = 'layers_configs' if 'layers_configs' in model_config['net']['config'] else 'layers'
        for layer in model_config['net']['config'][layers_key]:
            if 'condition_size' in layer:
                layer['condition_size'] = condition_size

    # Run training
    print(f"Running knob-conditioned training with configs:\n  Data: {DATA_CONFIG}\n  Model: {MODEL_CONFIG}\n  Learning: {LEARNING_CONFIG}\n  Output: {OUTDIR}")
    main(
        data_config=data_config,
        model_config=model_config,
        learning_config=learning_config,
        outdir=outdir,
        no_show=True,
        make_plots=True,
    )
    print(f"\nTraining complete. Results and model files are in: {OUTDIR}") 