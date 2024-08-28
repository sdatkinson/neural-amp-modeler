# NAM: Neural Amp Modeler (tweaks)
# MODDED Parameters/options

* Standard WaveNet is what I call HIGH and uses around 2x CPU compared to Standard Wavenet but the quality is superior. THIS IS NOT configured YET. USE LITE (my STANDARD) FOR THE TIME BEING.
* Lite WaveNet is equivalent to Stardard Wavenet's CPU usage (1x)

I don't like the way medium to high gain sounds with default settings. They start to sound the same and kind of what I believe many would say "transistor like". I think these settings sound much better. Aliasing sounding artifacts which I believe causes the "transistor like" sound are attenuated and also high frequency post-echos.

This repository handles training, reamping, and exporting the weights of a model.
For playing trained models in real time in a standalone application or plugin, see the partner repo,
[NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).

For documentation, check out the [ReadTheDocs](https://neural-amp-modeler.readthedocs.io).
