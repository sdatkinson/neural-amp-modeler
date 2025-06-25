# NAM: Neural Amp Modeler with Knob Conditioning

> ⚠️ **WORK IN PROGRESS** ⚠️
> 
> This project is in active development and many features are experimental. In particular:
> - Real-time knob control is not yet implemented in the plugin
> - Community contributions and assistance would be greatly appreciated
> - Please check the issues section for current development status and ways to help

This repository by [Dr. Ori Cohen](https://www.linkedin.com/in/cohenori/) extends the original [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) (created by Steven Atkinson) by adding support for modeling the entire range of an amp's controls through knob conditioning. While the original NAM allows modeling of guitar amps and effects at fixed settings, this enhanced version enables complete modeling of all amp control parameters.

> **Important Note**: This implementation currently only works with WaveNet architecture due to its inherent support for conditioning. While efforts have been made to maintain backward compatibility with the original NAM plugin and models, this is not guaranteed for all features and configurations.

## What is Knob Conditioning?

Knob conditioning is a systematic approach to capturing an amp's full range of tones. There are several approaches to recording the training data:

### 1. Isolated Knob Recording (Default Approach)
Record each knob's range independently while keeping all other knobs at their default position (typically 12 o'clock):

- Record the amp with only the Volume knob changing (1 to 10), all others at default
- Then record with only the Bass knob changing through its range, others at default
- Repeat for each knob (Treble, Mid, Gain, etc.)

With this approach, the WaveNet architecture learns how each knob independently affects the sound and can interpolate between the recorded positions for each knob, while other knobs remain at their default settings.

### 2. Full Combinatorial Recording (Advanced)
Record the amp with various combinations of knob settings to capture knob interdependencies:

- Record all possible combinations of knob positions
- This captures how knobs interact with each other (e.g., how Bass and Mid interact)
- Requires significantly more recordings (grows exponentially with number of knobs and positions)
- For example: 5 knobs with 10 positions each would require 100,000 recordings (10^5)

### 3. Hybrid Approach (Balanced)
Record strategic combinations of knob settings to capture some interactions without the full combinatorial explosion:

- Choose key positions for each knob (e.g., 9 o'clock, 12 o'clock, and 3 o'clock)
- Record combinations of these positions
- Much more manageable than full combinatorial (e.g., 5 knobs with 3 positions each = 243 recordings (3^5))
- Captures main knob interactions while keeping the dataset size reasonable
- Allows interpolation between these key positions

The choice between approaches depends on your needs and recording capabilities:
- Isolated approach: Most practical, but assumes knob independence
- Full combinatorial: Most accurate, but requires extensive recording sessions
- Hybrid approach: Good balance of interaction capture and practicality

## Key Features

- **Multi-Knob Support**: Model amps with multiple controls (gain, bass, mid, treble, etc.) at various settings
- **Continuous Control**: Train on discrete knob positions but interpolate between them during use
- **Complete Amp Modeling**: Instead of capturing just one "sweet spot" setting, model the entire character of an amp
- **WaveNet Architecture**: Leverages WaveNet's conditioning capabilities for knob parameter integration
- **Real-time Control**: *(Coming Soon)* Support for real-time knob adjustment in the NAM plugin is under development

## Technical Details

### Knob Conditioning Implementation
The conditioning mechanism works by feeding two pieces of information at each timestep of the WaveNet model:
1. **Knob Type**: One-hot encoded vector identifying which knob is being adjusted (e.g., [1,0,0] for Volume, [0,1,0] for Bass, etc.)
2. **Knob Level**: Float value representing the knob position (e.g., 0.5 for middle position)

This per-timestep conditioning allows the model to:
- Uniquely identify each knob type
- Process precise knob positions
- Learn the specific effect each knob has on the audio signal at that moment

For example, a Volume knob at 75% would be represented as:
```
Knob Type: [1,0,0,0] (one-hot for Volume)
Level: 0.75 (knob position)
```

This information is integrated into WaveNet's dilated convolution layers, allowing the model to adjust its processing based on the current knob configuration.

### Relation to Original WaveNet
This conditioning approach is directly inspired by and similar to WaveNet's original speaker conditioning mechanism, where different speakers were identified using one-hot encodings. In our case, we extend this concept by:
- Using knob types instead of speaker identities
- Adding a continuous level parameter for each knob
- Leveraging the same proven conditioning pathways in the WaveNet architecture

This adaptation of WaveNet's built-in conditioning capabilities ensures robust and reliable knob parameter integration.

## Development Status

This is an experimental project that needs community support to reach its full potential. Key areas where help is needed:

1. **Plugin Integration**: Implementing real-time knob control in the NAM plugin
2. **Performance Optimization**: Ensuring efficient processing for real-time parameter changes
3. **Testing**: Validating the model across different amp types and knob configurations
4. **Documentation**: Improving guides and examples for different recording approaches

If you're interested in contributing, please check the issues section or reach out directly.

## How It Works

The system uses knob conditioning to teach the neural network how different control settings affect the amp's sound. This is achieved through:

1. **Data Collection**: Record the amp at various knob settings
2. **Dataset Configuration**: Create a JSON file mapping recordings to knob settings
3. **Training**: The model learns to emulate the amp across all recorded settings using WaveNet architecture
4. **Interpolation**: Once trained, the model can interpolate between learned settings

## Configuration Files

The knob conditioning system uses several configuration files:

- `nam_full_configs/knobs/*.json`: Define knob dataset splits and configuration
- `data/knob_dataset.json`: Maps input/output audio files to their knob settings
- `nam_full_configs/learning/default.json`: Training configuration for knob-conditioned models
- `nam_full_configs/models/wavenet.json`: WaveNet model configuration (required for knob conditioning)

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
   # Example training command - Note: Must use WaveNet model
   python run_knob_conditioned_training.py \
     --data nam_full_configs/knobs/your_config.json \
     --model nam_full_configs/models/wavenet.json \
     --learning nam_full_configs/learning/default.json
   ```

3. **Use the Model** *(Limited Functionality)*:
   - Model training and export works, but real-time control is under development
   - Plugin integration for knob control is a work in progress
   - Community help is welcome to implement these features

> **Note**: While the training pipeline is functional, real-time parameter control in the plugin is still being developed. Please consider contributing to help implement these features!

## Documentation

For detailed documentation on the original NAM system, see [ReadTheDocs](https://neural-amp-modeler.readthedocs.io).

For knob conditioning specifics, check the following files:
- `nam/data.py`: Implementation of knob-conditioned datasets
- `nam/load_knob_data.py`: Knob dataset loading utilities
- `nam_full_configs/models/wavenet.json`: WaveNet configuration example
- Example configurations in `nam_full_configs/`
