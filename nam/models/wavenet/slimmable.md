# Slimmable Class Hierarchy

Class diagram for slimmable training in `nam/models/wavenet/__init__.py`.

## Hierarchy Diagram

```
nn.Module
├── _Slimmable (ABC, nn.Module)
│   │   Mixin: manages _adjust_size ratio (0–1); adjust_to(), adjust_to_random(), context_adjust_to_random()
│   │
│   ├── _WaveNet (_Slimmable, nn.Module)
│   │   Top-level model; propagates adjust_to() to child modules
│   │
│   └── _SlimmableConv1dBase (nn.Conv1d, _Slimmable)
│       Base: abstract _get_adjusted_weight_and_bias(); forward() uses sliced w,b when adjusted
│       │
│       ├── _SlimmableRechannelIn
│       │   Rechannel into layer array. First layer: slice out only; later: slice in and out.
│       │   _max_adjust_size = out_channels
│       │
│       ├── _SlimmableConvLayer
│       │   Layer conv: channels → mid_channels. Gated: mid=2*ch, slice w[:2*adj,:adj,:]
│       │   _max_adjust_size = in_channels
│       │
│       ├── _SlimmableInputMixer
│       │   Condition → mid_channels. Slice output only.
│       │   _max_adjust_size = out_channels
│       │
│       ├── _Slimmable1x1
│       │   1×1 in residual path. Slice both in and out (in_channels == out_channels).
│       │   _max_adjust_size = in_channels (= out_channels)
│       │
│       └── _SlimmableHeadRechannel
│           channels → 1. Slice input only.
│           _max_adjust_size = in_channels
```

## Data Flow (Slimmable Mode)

```
_LayerArray (slimmable=True)
│
├── _rechannel:     _SlimmableRechannelIn (input_size → channels)
│
├── _layers[i]:     _Layer
│   ├── _conv:      _SlimmableConvLayer (channels → mid_channels; gated)
│   ├── _input_mixer: _SlimmableInputMixer (condition_size → mid_channels)
│   └── _layer1x1:  _Slimmable1x1 (bottleneck → channels)
│
└── _head_rechannel: _SlimmableHeadRechannel (channels → ``head.out_channels``; kernel_size 1 only for slimmable)
```

## Slice Semantics (weight tensor w)

| Class                   | Slicing                               | Notes                         |
|-------------------------|---------------------------------------|-------------------------------|
| _SlimmableRechannelIn   | first: w[:adj,:,:]; later: w[:adj,:adj,:] | Rechannel into layer array   |
| _SlimmableConvLayer     | gated: w[:2*adj,:adj,:], b[:2*adj]; else: w[:adj,:adj,:] | Dilated conv, optional gate |
| _SlimmableInputMixer    | w[:adj,:,:], b[:adj]                  | Condition mixer              |
| _Slimmable1x1           | w[:adj,:adj,:], b[:adj]               | Residual path 1×1            |
| _SlimmableHeadRechannel | w[:,:adj,:], b unchanged              | Head: channels → 1           |

## Helper

- `_ratio_to_channels(ratio, max_size)` → `1 + min(floor(ratio * max_size), max_size - 1)` (min 1 channel)
