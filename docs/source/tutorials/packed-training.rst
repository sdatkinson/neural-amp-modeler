Packed training
===============

Packed training trains several compatible WaveNet submodels inside one larger
masked WaveNet. Each submodel makes its own prediction for the same target
audio, and training computes the loss for each prediction against that target
before summing the losses.

For A2, the simplified GUI and Colab trainers use a fixed packed WaveNet 
configuration automatically. This page covers configuring packed training 
directly with the full trainer.

Packed training is a slimmable NAM training method, but it is different from
the channel-slicing slimmable WaveNet method. Packed training keeps the
submodels independent during training by using block-diagonal masked weights.
The exported ``model.nam`` is also not a packed inference graph: export extracts
ordinary ``WaveNet`` models and stores them in a ``SlimmableContainer``.

When to use packed training
---------------------------

Use packed training when you want one ``model.nam`` container with several
WaveNet sizes, and the submodels share the same temporal architecture but use
different channel counts. This is useful when training several related WaveNet
models separately would repeat much of the same work.

Packed training is currently intended for compatible mono WaveNet models. If
the submodels need different kernels, dilations, activation types, conditioning
paths, heads, or grouped/FiLM features, train them separately for now.

Configuration
-------------

Packed training uses the normal full trainer command. Put ``"PackedWaveNet"``
in the model config and run ``nam-full`` the same way as ordinary full training:

.. code-block:: console

    $ nam-full path/to/data.json path/to/model.json path/to/learning.json path/to/outputs

No extra command-line flag is needed. The trainer chooses
``PackedLightningModule`` from the model config.

The public example config is
`nam_full_configs/models/wavenet_packed.json <https://github.com/sdatkinson/neural-amp-modeler/blob/main/nam_full_configs/models/wavenet_packed.json>`_.
An abbreviated version looks like this:

.. code-block:: json

    {
      "net": {
        "name": "PackedWaveNet",
        "config": {
          "submodels": [
            {
              "name": "small",
              "config": {
                "layers_configs": [
                  {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 3,
                    "head": {"out_channels": 1, "kernel_size": 1, "bias": true},
                    "kernel_size": 6,
                    "dilations": [1, 5, 29, 97, 227],
                    "activation": "LeakyReLU"
                  }
                ],
                "head": null,
                "head_scale": 0.01
              }
            },
            {
              "name": "large",
              "config": {
                "layers_configs": [
                  {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 8,
                    "head": {"out_channels": 1, "kernel_size": 1, "bias": true},
                    "kernel_size": 6,
                    "dilations": [1, 5, 29, 97, 227],
                    "activation": "LeakyReLU"
                  }
                ],
                "head": null,
                "head_scale": 0.01
              }
            }
          ],
          "export": {
            "container_max_values": "uniform"
          }
        }
      }
    }

The ``submodels`` list names each exported model and gives each one an ordinary
WaveNet ``config``. Channel counts such as ``channels`` may differ between
submodels. Temporal settings such as ``kernel_size`` and ``dilations`` must
match.

The optional ``export.container_max_values`` setting controls the
``max_value`` thresholds written into the exported ``SlimmableContainer``. Use
``"uniform"`` to spread the thresholds evenly across the submodels, or provide
a sorted list with one value per submodel. The final threshold is written as
``1.0``.

Compatibility requirements
--------------------------

Packed submodels must be structurally compatible:

* The submodels must have the same number of layer arrays.
* Corresponding layer arrays must have the same number of layers.
* Corresponding layer arrays must use the same kernel sizes and dilations.
* Activation configuration must match across corresponding layers.
* ``head_scale`` must match across submodels.
* Condition sizes must match, and the current implementation expects mono
  conditioning audio.
* Layer-array head settings must line up across arrays. In multi-array models,
  each array's input and head path must match the previous array's channel and
  head outputs.
* Layer-array head kernel sizes and bias flags must match.
* ``layer_1x1`` and ``head_1x1`` active flags must match.

Channel counts may differ. That is the usual reason to use packed training.

Unsupported combinations
------------------------

Packed training currently does not support:

* ``condition_dsp``.
* Top-level WaveNet ``head`` configs.
* FiLM.
* Grouped convolutions.
* Grouped ``layer1x1`` or ``head1x1``.
* Paired or gated activations.
* Packed training plus channel-slicing slimmable WaveNet settings inside the
  same submodel.
* Multi-channel input.
* Multi-channel conditioning audio.
* Multi-channel output beyond one output channel per packed submodel.

Training behavior
-----------------

During training, the packed model returns predictions shaped like
``(batch, submodel, time)``. The trainer computes the configured training loss
for each submodel prediction against the same target audio and sums the losses.

Validation logs aggregate metrics such as ``val_loss``, ``ESR``, and ``MRSTFT``
as well as per-submodel metrics such as ``val_loss_packed_0``, ``ESR_packed_0``,
and ``MRSTFT_packed_0``. When
validation is available, the trainer may also save per-submodel best
checkpoints named ``packed_best_submodel_<i>.ckpt``.

Output files
------------

The output directory contains the normal full-training artifacts, including
copied configs, Lightning checkpoints, and optional comparison plots. Packed
training adds packed-specific export behavior:

* ``model.nam`` has architecture ``SlimmableContainer``.
* The container embeds complete ordinary ``WaveNet`` models.
* The packed training model itself is not the runtime format.
* ``packed_best.json`` is written when per-submodel validation checkpoints are
  available.
* ``packed_best_submodel_<i>.ckpt`` files may be written for the best checkpoint
  of each packed submodel.

If per-submodel best checkpoints are available at final export, the container
uses them for the corresponding extracted submodels. Otherwise final export
falls back to the current or aggregate checkpoint behavior used by the trainer.

Troubleshooting
---------------

Compatibility validation errors usually mean the submodel configs differ in
temporal architecture or use a feature listed above as unsupported. Check the
corresponding layer arrays first: depth, kernel sizes, dilations, activation
configuration, ``head_scale``, head path, and mono input/output settings are
the most common places to look.

Packed training can use more registered parameters than the sum of the
individual submodels because the packed tensors include masked off-block
weights. Those masked weights are kept out of the independent submodel paths
during training and are not exported as a packed runtime graph.
