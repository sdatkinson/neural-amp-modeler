# NAM: neural amp modeler

This repository handles training, reamping, and exporting the weights of a model.
For playing trained models in real time in a standalone application or plugin, see the partner repo,
[NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

## How to use (Google Colab)

If you don't have a good computer for training ML models, you use Google Colab to train
in the cloud using the pre-made notebooks under `bin\train`.

For the very easiest experience, simply go to
[https://colab.research.google.com/github/sdatkinson/neural-amp-modeler/blob/main/bin/train/easy_colab.ipynb](https://colab.research.google.com/github/sdatkinson/neural-amp-modeler/blob/main/bin/train/easy_colab.ipynb) and follow the
steps!

For a little more visibility under the hood, you can use [colab.ipynb](https://colab.research.google.com/github/sdatkinson/neural-amp-modeler/blob/main/bin/train/colab.ipynb) instead.

**Pros:**

- No local installation required!
- Decent GPUs are available if you don't have one on your computer.

**Cons:**

- Uploading your data can take a long time.
- The session will time out after a few hours (for free accounts), so extended
  training runs aren't really feasible. Also, there's a usage limit so you can't hang
  out all day. I've tried to set you up with a good model that should train reasonably
  quickly!

## How to use (Local)

Alternatively, you can clone this repo to your computer and use it locally.

### Installation

Installation uses [Anaconda](https://www.anaconda.com/) for package management.

For computers with a CUDA-capable GPU (recommended):

```bash
conda env create -f environment_gpu.yml
```

Otherwise, for a CPU-only install (will train much more slowly):

```bash
conda env create -f environment_cpu.yml
```

Then activate the environment you've created with

```bash
conda activate nam
```

### Things you can do

Here are the primary ways this is meant to be used:

#### Train a model

You'll need at least two mono wav files: the input (DI) and the amped sound (without the cab).
You can either record enough to have a training and validation set in the same file and
split the file, or you can use 4 files (input/output for train/test).
Also, you can provide _multiple_ file pairs for training (or validation).

For the first option, Modify `bin/train/inputs/config_data_single_pair.json` to point at the audio files, and set the
start/stop to the point (in samples) where the training segment ends and the validation
starts.
For the second option, modify and use `bin/train/inputs/config_data_two_pairs.json`.

Then run:

```bash
python bin/train/main.py \
bin/train/inputs/config_data.json \
bin/train/inputs/config_model.json \
bin/train/inputs/config_learning.json \
bin/train/outputs/MyAmp
```

#### Run a model on an input signal ("reamping")

Handy if you want to just check it out without going through the trouble of building the
plugin.

For example:

```bash
python bin/run.py \
path/to/source.wav \
path/to/config_model.json \
path/to/checkpoints/epoch=123_val_loss=0.000010.ckpt \
path/to/output.wav
```

#### Export a model (to use with [the plugin](https://github.com/sdatkinson/iPlug2))

Let's get ready to rock!

```bash
python bin/export.py \
path/to/config_model.json \
path/to/checkpoints/epoch=123_val_loss=0.000010.ckpt \
path/to/exported_models/MyAmp
```

Then point the plugin at the exported model directory and you're good to go!

## Advanced usage

The model architectures and cofigurations in `bin/train/inputs/models` should work plenty well out of the box.
However, feel free to play around with it; sometimes some tweaks can help improve performance.

Also, you can train for shorter or longer.
1000 epochs is typically overkill, but how little you can get away with depends on the model you're using.
I recommend watching the checkpoints and keeping an eye out for when the ESR drops below
0.01--usually it'll sound pretty good by that point.
