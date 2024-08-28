# NAM: Neural Amp Modeler (tweaks)
# MODDED Parameters/settings

* Standard WaveNet with my settings uses around 2x CPU compared to Standard Wavenet but the quality is superior in my opinion.
  THIS IS NOT CONFIGURED YET. USE LITE (my STANDARD) FOR THE TIME BEING.

* Lite WaveNet is equivalent to Stardard Wavenet's CPU usage (1x) but the quality is superior in my, once again, opinion.

I wanted to improve the already very good default settings but make them extremelly good. The default settings unfortunately start to sound the same and kind of what I believe many would say "transistor like" and becomes more obvious the more gain that is modeled. I think my settings sound much better. Aliasing sounding artifacts which I believe causes the "transistor like" sound are attenuated and also high frequency post-echos.

In the future I will try to add my own training file as an accepted input. It is built upon v3_0_0.wav so the first part of the file is 1:1 copy and is followed by mix of test tones, sweeps, noise, percussion, guitar, bass guitar and some more test tones to more easily see if the signal is inverted.


This repository handles training, reamping, and exporting the weights of a model.
For playing trained models in real time in a standalone application or plugin, see the partner repo,
[NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

For documentation, check out the [ReadTheDocs](https://neural-amp-modeler.readthedocs.io).
