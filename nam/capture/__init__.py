"""
Guided parametric capture: plan knob settings, reamp through an audio interface, and
keep a resumable project plus a training-ready ``data.json`` up to date.

Submodules are imported explicitly (``nam.capture.params``, ``nam.capture.planner``,
...) rather than re-exported here: several of them pull in heavy dependencies (torch,
sounddevice, the training stack) that the desktop app defers until needed.
"""
