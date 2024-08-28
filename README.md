# NAM: Neural Amp Modeler (tweaks)
# MODDED Parameters/settings

I wanted to improve the already very good default settings but make them extremelly good. The default settings unfortunately start to sound the same and kind of what I believe many would say "transistor like" and becomes more obvious the more gain that is modeled. I think my settings sound much better. Aliasing sounding artifacts which I believe causes the "transistor like" sound are attenuated and also high frequency post-echos.

* Standard WaveNet with my settings uses around 2x CPU compared to Standard Wavenet but the quality is superior in my opinion.

* Lite WaveNet is equivalent to Stardard Wavenet's CPU usage (1x) but the quality is once again, in my opinion, superior.

I train most models around 500 epochs. Some easy ones without much sag/compression only needs 300. The tough ones with sag/compression and a lot of character sometimes train up to 800 epochs.

In the future I will try to add my own training file as an accepted input. It is built upon v3_0_0.wav so the first part of the file is 1:1 copy and is followed by mix of test tones, sweeps, noise, percussion, guitar, bass guitar and some more test tones to more easily see if the signal is inverted.


--------------------------------------------------------------------------------
This repository handles training, reamping, and exporting the weights of a model.
For playing trained models in real time in a standalone application or plugin, see the partner repo,
[NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

For documentation, check out the [ReadTheDocs](https://neural-amp-modeler.readthedocs.io).
