.. _nam-file-spec:

``.nam`` file specification and change log
==========================================

The current specification
-------------------------

``.nam`` files are currently able to be parsed as JSON files. The outermost data 
structure is a dictionary.

There are a few keys you should expect to find with the following values:

* ``"version"``: A string stating the version of the file. It follows semantic
  versioning.
* ``"architecture"``: The high-level architecture of the model (usually either 
  "WaveNet" or "LSTM", though any string corresponding to a model class name 
  defined in the ``nam`` package is valid. Also, if you've made a new 
  architecture, you would give it a new name here. The convention is to use the 
  class name.)
* ``"config"``: A dictionary architecture-specific details (e.g. number of 
  layers, etc).
* ``"weights"``: a list of float-type numbers that are the weights (parameters) 
  of the model. How they map into the model is architecture-specific. Looking at
  ``._export_weights()`` will usually tell you what you need to know (e.g. for 
  ``WaveNet`` at
  `wavenet.py <https://github.com/sdatkinson/neural-amp-modeler/blob/cb100787af4b16764ac94a2edf9bcf7dc5ae59a7/nam/models/wavenet.py#L428>`_ 
  and ``LSTM`` at
  `recurrent.py <https://github.com/sdatkinson/neural-amp-modeler/blob/cb100787af4b16764ac94a2edf9bcf7dc5ae59a7/nam/models/recurrent.py#L317>`_.)

There are also some optional keys that ``nam`` may use:

* ``"sample_rate"``: The (possibly non-integer) sample rate of the data that the
  model expects to see, in samples/sec (Hz) If not present, one should generally
  assume the sample rate was 48kHz.
* ``"metadata"``: A dictionary with key-value pairs of information about the 
  model. The following may be used by this package:

  * ``"date"``: a dictionary with the ``"year"``, ``"month"``, ``"day"``, 
    ``"hour"``, ``"minute"``, and ``"second"`` at which the model was exported 
    (all integer-type).
  * ``"name"``: The name of the model (could be used as the display name by plugins.)
  * ``"modeled_by"``: Who made the model
  * ``"gear_make"``: Make of the gear (E.g. Fender)
  * ``"gear_model"``: Model of the gear (E.g. Deluxe Reverb)
  * ``"gear_type"``: What kind of gear this is a model of. Options are 
    ``"amp"``, ``"pedal"``, ``"pedal_amp"``, ``"amp_cab"``, ``"amp_pedal_cab"``,
    ``"preamp"``, and ``"studio"``.
  * ``"tone_type"``: How the model sounds. Options are ``"clean"``, 
    ``"overdrive"``, ``"crunch"``, ``"hi_gain"``, and ``"fuzz"``.
  * ``"training"``: A dictionary containing information about training (*Only 
    when the simplified trainers are used.*)
  * ``"input_level_dbu"``: The level being input to the gear, in dBu, corresponding to a
    1kHz sine wave with 0dBFS peak.
  * ``"output_level_dbu"``: The level, in dBu, of a 1kHz sine wave that achieves 0dBFS
    peak when input to the interface that's recording the output of the gear being
    modeled.
  * ``"loudness"``: How loud the model is, in dB, when fed a standardized input
    (written by the trainer at export). Players use it to keep a steady volume
    when switching models, typically as ``gain_dB = -18 - loudness``. The input
    is a 1.75 s train of seven instrument-shaped noise bursts spanning -39 to
    -15 dBFS RMS; the model's output is K-weighted (ITU-R BS.1770), the level of
    each burst is measured in dB, and the burst levels are averaged in dB with a
    fixed +3 dB calibration offset. Measuring over a range of input levels makes
    the value insensitive to how hard the player drives the model. Files
    exported by versions of ``nam`` on v0.13.0 or older used the flat RMS of
    1 s of pink noise at -15 dBFS instead; the two agree in their median but
    differ per model by up to roughly +-10 dB. See
    `tone-3000/loudness-research <https://github.com/tone-3000/loudness-research>`_
    for the evaluation behind the definition.
  * ``"gain"``: A heuristic between 0 and 1 for how much gain/compression the
    model appears to have (0 is linear), derived from how the output level grows
    as the standardized input is scaled.
  * ``"weights_hash"``: identifies the weights that are actually run, independent
    of the rest of the file, so editing metadata does not change it; two files with
    the same ``weights_hash`` are the same model. Written by the trainer at export,
    after the post hooks, as ``"nam-content-v1:"`` followed by 64 lowercase hex
    characters::

      weights_hash = "nam-content-v1:" + hex(SHA-256(
          "nam-content-v1" 0x00
          for each leaf model, depth-first:
              u64le(N) || f32le(w[0]) || ... || f32le(w[N-1])
      ))

    Each weight is rounded to a 32-bit float, with ``-0`` written as ``+0``; ``N`` is
    the leaf's weight count. Models that contain other models have empty
    ``"weights"`` and hash their children: a ``"SlimmableContainer"`` from the
    smallest ``max_value`` to the largest, a ``"Sequential"`` model in processing
    order. A file with a leaf that has no weights, a malformed container, or a
    weight that isn't a finite float32 gets no ``weights_hash``.
    ``nam.models.content_hash.weights_hash(model_dict)`` computes it for any parsed
    ``.nam`` file, and ``tests/resources/content_hash_test_vectors.json`` lists
    models with their expected hash for checking other implementations.


Sequential models
-----------------

``"Sequential"`` models compose complete child NAM models in processing order.
They use the existing file version and top-level envelope. The wrapper has an
empty ``"weights"`` array because its parameters belong to its children::

  {
    "version": "0.7.0",
    "architecture": "Sequential",
    "config": {
      "models": [
        {"version": "0.7.0", "architecture": "WaveNet", "config": {}, "weights": [], "sample_rate": 48000},
        {"version": "0.7.0", "architecture": "Linear", "config": {}, "weights": [], "sample_rate": 48000}
      ]
    },
    "weights": [],
    "sample_rate": 48000
  }

Every entry in ``config.models`` is a complete NAM model with its own
architecture, configuration, and weights. The top-level and child sample rates
must agree.

Sequential files exported before Core support was completed used bare child
configs and concatenated top-level weights. Because those files omitted the
child architectures, they cannot be interpreted reliably and are not supported
by this canonical format.


Change log
----------

v0.5
^^^^

v0.5.4
""""""

Introduced in ``neural-amp-modeler`` `version 0.10.0 <https://github.com/sdatkinson/neural-amp-modeler/releases/tag/v0.10.0>`_.

* Add ``"input_level_dbu"`` and ``"output_level_dbu"`` fields under ``"metadata"``.

v0.5.3
""""""

Introduced in ``neural-amp-modeler`` `version 0.9.0 <https://github.com/sdatkinson/neural-amp-modeler/releases/tag/v0.9.0>`_.

* Add ``"training"`` field under ``"metadata"`` whose contents follow the
  ``TrainingMetadata`` 
  `Pydantic model <https://github.com/sdatkinson/neural-amp-modeler/blob/cb100787af4b16764ac94a2edf9bcf7dc5ae59a7/nam/train/metadata.py#L84>`_. (`#420 <https://github.com/sdatkinson/neural-amp-modeler/pull/420>`_)

v0.5.2
""""""

Version corresponding to ``neural-amp-modeler`` 
`version 0.5.2 <https://github.com/sdatkinson/neural-amp-modeler/releases/tag/v0.5.2>`_.
TODO more info.
