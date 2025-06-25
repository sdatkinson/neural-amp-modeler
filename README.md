# NAM: Neural Amp Modeler with Knob Conditioning

This repository was forked from [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) by Steven Atkinson. While the original NAM allows modeling of guitar amps and effects at fixed settings, this fork adds support for modeling the entire range of an amp's controls through knob conditioning.

## Key Features

- **Multi-Knob Support**: Model amps with multiple controls (gain, bass, mid, treble, etc.) at various settings
- **Continuous Control**: Train on discrete knob positions but interpolate between them during use
- **Complete Amp Modeling**: Instead of capturing just one "sweet spot" setting, model the entire character of an amp
- **Real-time Control**: Adjust virtual knobs in real-time through the [NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin)

## How It Works

The system uses knob conditioning to teach the neural network how different control settings affect the amp's sound. This is achieved through:

1. **Data Collection**: Record the amp at various knob settings
2. **Dataset Configuration**: Create a JSON file mapping recordings to knob settings
3. **Training**: The model learns to emulate the amp across all recorded settings
4. **Interpolation**: Once trained, the model can interpolate between learned settings

## Configuration Files

The knob conditioning system uses several configuration files:

- `nam_full_configs/knobs/*.json`: Define knob dataset splits and configuration
- `data/knob_dataset.json`: Maps input/output audio files to their knob settings
- `nam_full_configs/learning/default.json`: Training configuration for knob-conditioned models

### Example Knob Dataset Format
```json
{
  "entries": [
    {
      "input_file": "data/input_1.wav",
      "output_file": "data/output_1.wav",
      "knob_type": "gain",
      "knob_level": 0.5
    },
    // ... more entries for different knob combinations
  ]
}
```

## Usage

1. **Record Your Amp**:
   - Record input/output pairs at different knob settings
   - Document the settings in your dataset JSON file

2. **Configure Training**:
   ```bash
   # Example training command
   python run_knob_conditioned_training.py \
     --data nam_full_configs/knobs/your_config.json \
     --model nam_full_configs/models/convnet.json \
     --learning nam_full_configs/learning/default.json
   ```

3. **Use the Model**:
   - Load the trained model in the NAM plugin
   - Control the virtual knobs in real-time

## Documentation

For detailed documentation on the original NAM system, see [ReadTheDocs](https://neural-amp-modeler.readthedocs.io).

For knob conditioning specifics, check the following files:
- `nam/data.py`: Implementation of knob-conditioned datasets
- `nam/load_knob_data.py`: Knob dataset loading utilities
- Example configurations in `nam_full_configs/`
