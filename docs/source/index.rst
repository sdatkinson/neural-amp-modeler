Welcome to ``neural-amp-modeler``'s documentation!
==================================================

``neural-amp-modeler`` is a Python package for creating neural network models of
your guitar (bass, etc) gear. It works by using two audio files--an input "DI"
file as well as an output "reamp" file, showing how the gear responds to
different incoming signals.

Model limitations
-----------------

The warning printed by the simplified trainers about "time-based effects"
(chorus, delay, reverb) describes an assumption made by those trainers, not a
hard limit of NAM models. A linear impulse response (``nam.models.linear.Linear``)
can represent a linear delay or reverb, and stacked WaveNet or LSTM models can
model other effects.

A WaveNet is time-invariant by construction, so it is not a good fit for a
pitch-shifter or octaver; an LSTM, or an extension, can model behaviour that
depends on time. Compressors are usually time-invariant but have history
dependence, which the simplified trainers are not designed to capture.

See `issue #525 <https://github.com/sdatkinson/neural-amp-modeler/issues/525>`_
for the maintainer's detailed explanation.

Contents
--------

.. toctree::
   :maxdepth: 1

   installation
   tutorials/main
   api
   model-file
