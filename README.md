# NAM: neural amp modeler

This repository handles training, reamping, and exporting the weights of a model.
For the code to create the plugin with a trained model, see my
[iPlug2 fork](https://github.com/sdatkinson/iPlug2).

## How to use (Google Colab)

If you don't have a good computer for training ML models, you can run the
notebook located at `bin/train/colab.ipynb` in the cloud using Google Colab--no
local installation required!

Go to [colab.research.google.com](https://colab.research.google.com), select the 
"GitHub" tab, paste in the URL for the notebook 
(https://github.com/sdatkinson/neural-amp-modeler/blob/dev/bin/train/colab.ipynb), 
and go!

**Pros:**
* No local installation required!
* Decent GPUs are available if you don't have one on your computer.

**Cons:**
* Uploading your data can take a long time.
* The session will time out after an hour or so (for free accounts), so extended 
  training runs aren't really feasible.  I've tried to set you up with a good
  model that should train quickly!

## How to use (Local)

Alternatively, you can clone this repo to your computer and use it locally.

### Installation
```bash
conda env create -f environment.yml
conda activate nam
pip install .
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

You'll want the `HardCodedModel.h` to paste over into the plugin source (i.e. [here](https://github.com/sdatkinson/iPlug2/blob/5a0f533f7a9e4ee691da26adb2a38d87905e87fe/Examples/NAM/HardCodedModel.h)).

## Advanced usage

The model architectures and cofigurations in `bin/train/inputs/models` should work plenty well out of the box. 
However, feel free to play around with it; sometimes some tweaks can help improve performance.

Also, you can train for shorter or longer.
1000 epochs is typically overkill, but how little you can get away with depends on the model you're using.
I recommend watching the checkpoints and keeping an eye out for when the ESR drops below 0.01--usually it'll
sound pretty good by that point.
