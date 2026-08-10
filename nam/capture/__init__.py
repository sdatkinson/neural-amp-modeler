"""
Guided parametric capture: plan knob settings, reamp through an audio interface, and
keep a resumable project plus a training-ready ``data.json`` up to date.

Submodules are imported explicitly (``nam.capture.params``, ``nam.capture.planner``,
...) rather than re-exported here: several of them pull in heavy dependencies (torch,
sounddevice, the training stack) that the desktop app defers until needed.
"""

# The capture app's own version, hand-maintained. Separate from ``nam.__version__``,
# which is generated from version-control tags and describes the whole package. This
# exists to date a project's behaviour: it is stamped into ``capture_project.json`` when
# a project is created and never changed afterwards, so a project always says which
# version's rules it started under.
CAPTURE_APP_VERSION = "1.1.0"

# The version raw recordings (``captures_raw/``) started being saved in. Fixed forever at
# the version that introduced them, while CAPTURE_APP_VERSION moves on: a project created
# before this has captures with no raw recordings behind them, which is worth saying
# plainly rather than leaving as a mystery about a half-empty folder.
RAW_RECORDING_SINCE_VERSION = "1.1.0"
