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

### Train models (GUI)
After installing, you can open a GUI trainer by running

```bash
nam
```

from the terminal.

### Train models (Python script)
For users looking to get more fine-grained control over the modeling process, 
NAM includes a training script that can be run from the terminal, e.g.:

```bash
python bin/train/main.py \
bin/train/inputs/config_data.json \
bin/train/inputs/config_model.json \
bin/train/inputs/config_learning.json \
bin/train/outputs/MyAmp
```

where `config_data.json` contains the information about the data you're training
on, `config_model.json` contains information about the model architecture that
is being trained, and `config_learning.json` contains information about the 
training run itself (e.g. number of epochs).
You'll need to configure the data JSON to the specifics of the data you're 
training on. The others may work for your needs out-of-the-box with no 
modification.

Since NAM uses [PyTorch Lightning](https://lightning.ai/pages/open-source/) 
under the hood as a modeling framework, many of the configuration options that
are passed to its componenets can be configured from the data/model/learning
JSONs.

#### Export a model (to use with [the plugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin))
Exporting the trained model to a `.nam` file for use with the plugin can be done
with:

```bash
python bin/export.py \
path/to/config_model.json \
path/to/checkpoints/epoch=123_val_loss=0.000010.ckpt \
path/to/exported_models/MyAmp
```

Then, point the plugin at the exported `model.nam` file and you're good to go!

### Other utilities

#### Run a model on an input signal ("reamping")

Handy if you want to just check it out without needing to use the plugin:

```bash
python bin/run.py \
path/to/source.wav \
path/to/config_model.json \
path/to/checkpoints/epoch=123_val_loss=0.000010.ckpt \
path/to/output.wav
```
