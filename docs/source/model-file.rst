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
