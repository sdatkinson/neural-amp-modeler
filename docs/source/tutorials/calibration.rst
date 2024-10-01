How to determine the calibration levels of your recording setup
===============================================================

Background
----------

With the arrival of highly-accurate modeling technologies like NAM, there's been
an interest in ensuring that the gain staging of the signal entering and leaving
models is accurate to what one would experience when plugging into the source
analog gear.

As of version 0.10.0, the NAM defines a pair of metadata fields, 
``"input_level_dbu"`` and ``"output_level_dbu"``, that document the relationship
between the *digital* and *analog* signal strengths. Specifically, these fields
record the *analog* signal strength (in dBu) that corresponds to the loudest
signal that can be represented *digitally*. With these two values recorded, one
can calibrate the gain staging of the digital models to accurately reflect the
behavior of the source in its native analog realm.

This tutorial explains how to take the measurements to fill in these metadata.

.. note:: As with all metadata, recording the calibration levels of the
    recording is optional. If you don't want to do this, you can leave them
    blank. The mdoels will still work in any plug-in that supports playing NAMs,
    but the gain-staging may not be accurate to the source gear.

Tools needed
------------

In addition to your reamping gear (interface, reamp box, DI box, load 
box/microphone, cables), you will need a
`digital multimeter <https://en.wikipedia.org/wiki/Multimeter>`_ to measure the
analog signal being sent and returned to your interface.

Procedure
---------

First, set up your gear as you would normally for reamping:

* Mute the inputs to your interface in your DAW.
* Connect the output of your interface to the input of your gear via the reamp
  box.
* Connect the output from your gear (or load box, or microphone) to the input of
  your interface.
* Turn the output from your interface to its maximum value.
* Begin reamping and set the gain on the return input to your interface so that
  no clipping occurs.

[TODO picture of reamp setup.]

.. note:: My advice is to set the reamping send level as high as your gear will
  allow. If it's too low, then the model won't see any examples of the gear
  distorting under a very hot signal and may not predict it accurately. You 
  don't need to worry if it's exactly accurate to how loud a guitar is--that's 
  the purpose of calibration--what's important is that the model has trained on 
  examples that are at least as loud as (preferably even louder than!) how the 
  model will be used in practice.

Next, measure the send level. To do this, play a sine wave with 1kHz frequency
and 0dBFS peak amplitude. Some plug-ins can do tone generation, or else you can
just loop this 1-second file: 
`sine 1k.wav <https://drive.google.com/file/d/18y53y4yi_QEUundLlBZsjdY_OeytC6y1/view?usp=drive_link>`_. 
Unplug your cable from the gear you are reamping and measure the RMS voltage
across its tip and sleeve.

[TODO picture of multimeter]

Convert the RMS voltage to dBu using the formula:

.. math::

   \text{dBu} = 20 \times \log_{10}\left(\frac{V_{\text{RMS}}}{0.7746}\right)

Alternatively, the bottom of this page has a table of pre-computed values you
can reference. This is the value you will provide to the send level field in the
trainer:

[TODO picture of GUI trainer]

[TODO picture of Colab trainer]

Next, measure the return level. TODO

The return level calibration isn't the same as the manufacturer specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As of this writing (September 2024), users have started to become familiar with
manufacturers' specifications for the input calibration to their interfaces 
(e.g. many tutorials on
`YouTube <https://www.youtube.com/results?search_query=plug-in+input+level>`_)

.. image:: media/calibration/youtube-input-level.png
    :scale: 50 %

The return level will not necessarily match the manufacturer's specification for
your interface's input level (e.g. for the 
`Focusrite Scarlett 2i2 4th Gen <https://downloads.focusrite.com/focusrite/scarlett-4th-gen/scarlett-2i2-4th-gen>`_,
the maximum input level at minimum gain for the instrument inputs is 12 dBu). 
The reason is that you may not be recording at minimum gain. As of writing this
(September 2024), common practice is to *normalize* the output level to a common
digital level so that different gear is all the same (digital) loudness. This
can be helpful for amp models where the output level of the model will
correspond to the level that will be mixed. However, for gear like pedals that 
will be fed as input to another piece of gear like an amp, this would result in 
the incorrect level being outputted from the pedal to the input of the amp.



A note on updating old model files
----------------------------------

If you know the levels from a model that you made previously, you can update 
old files to include this metadata. ``.nam`` files conform to the JSON file 
format and can be edited as plain text. If you want to do this, you can make a 
new model and use it as a reference for how to add the new metadata fields to 
your old files. Look for the fields ``"input_level_dbu"`` and 
``"output_level_dbu"`` in the new file and copy them to your old file in the 
corresponding location, changing the nubmers as necessary. As always, it's 
recommended to save a backup of your file before you being editing it in case 
you make a mistake.


Appendix: Conversion table between RMS voltage and dBu
------------------------------------------------------

+----------------+------------------+
| RMS Voltage (V)| dBu              |
+================+==================+
| 0.8             | 0.0             |
+-----------------+-----------------+
| 0.9             | 1.0             |
+-----------------+-----------------+
| 1.0             | 2.0             |
+-----------------+-----------------+
| 1.1             | 3.0             |
+-----------------+-----------------+
| 1.2             | 4.0             |
+-----------------+-----------------+
| 1.3             | 4.5             |
+-----------------+-----------------+
| 1.4             | 5.0             |
+-----------------+-----------------+
| 1.5             | 6.0             |
+-----------------+-----------------+
| 1.6             | 6.5             |
+-----------------+-----------------+
| 1.7             | 7.0             |
+-----------------+-----------------+
| 1.8             | 7.5             |
+-----------------+-----------------+
| 1.9             | 8.0             |
+-----------------+-----------------+
| 2.1             | 8.5             |
+-----------------+-----------------+
| 2.2             | 9.0             |
+-----------------+-----------------+
| 2.3             | 9.5             |
+-----------------+-----------------+
| 2.4             | 10.0            |
+-----------------+-----------------+
| 2.6             | 10.5            |
+-----------------+-----------------+
| 2.7             | 11.0            |
+-----------------+-----------------+
| 2.9             | 11.5            |
+-----------------+-----------------+
| 3.1             | 12.0            |
+-----------------+-----------------+
| 3.3             | 12.5            |
+-----------------+-----------------+
| 3.5             | 13.0            |
+-----------------+-----------------+
| 3.7             | 13.5            |
+-----------------+-----------------+
| 3.9             | 14.0            |
+-----------------+-----------------+
| 4.1             | 14.5            |
+-----------------+-----------------+
| 4.4             | 15.0            |
+-----------------+-----------------+
| 4.6             | 15.5            |
+-----------------+-----------------+
| 4.9             | 16.0            |
+-----------------+-----------------+
| 5.2             | 16.5            |
+-----------------+-----------------+
| 5.5             | 17.0            |
+-----------------+-----------------+
| 5.8             | 17.5            |
+-----------------+-----------------+
| 6.2             | 18.0            |
+-----------------+-----------------+
| 6.5             | 18.5            |
+-----------------+-----------------+
| 6.9             | 19.0            |
+-----------------+-----------------+
| 7.3             | 19.5            |
+-----------------+-----------------+
| 7.7             | 20.0            |
+-----------------+-----------------+
