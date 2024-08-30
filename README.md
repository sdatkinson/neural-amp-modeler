# NAM: Neural Amp Modeler (tweaks)
# MODDED Parameters/settings

I wanted to improve the already very good default settings but make them extremelly good. The default settings unfortunately start to sound the same and kind of what I believe some would say "transistor like" and becomes more obvious the more gain that is modeled. I think my settings sound much better and "aliasing sounding" artifacts which I believe causes the "transistor like" sound are attenuated and also high frequency post-echos. Compare my HIGH or EXTREME to a model "generic" STANDARD model and listen to when tones/notes decay. Lower quality tend to bring up the low frequencies and distort them. Most of this is subjective but I have also listened to them and compared and played finished models. What is needed is an ABX type of comparison.

* STANDARD is what I call XTRM (EXTREME) and uses about ~?x CPU compared to official STANDARD Wavenet models but the quality is precisely EXTREME.

* LITE is what I call HIGH and uses ~2x CPU compared to official STANDARD Wavenet models but the quality is superior in my opinion.

* FEATHER is CPU wise equivalent (1x) to official STANDARD Wavenet models but the quality is, in my opinion, overall greatly improved.

I train most models around 500 epochs. Some easy ones without much sag/compression only needs 300. The tough ones with sag/compression and a lot of character sometimes train up between 500-800 epochs.

In the future I will try to add my own training file as an accepted input. It is built upon v3_0_0.wav so the first part of the file is 1:1 copy and is followed by mix of test tones, sweeps, noise, percussion, guitar, bass guitar and some more blips tones to more easily see if the signal is inverted.


--------------------------------------------------------------------------------
This repository handles training, reamping, and exporting the weights of a model.
For playing trained models in real time in a standalone application or plugin, see the partner repo,
[NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

For documentation, check out the [ReadTheDocs](https://neural-amp-modeler.readthedocs.io).
