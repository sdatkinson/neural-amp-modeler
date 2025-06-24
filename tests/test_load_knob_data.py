import os
from tempfile import TemporaryDirectory
from nam.generate_knob_data import generate_knob_data
from nam.load_knob_data import load_knob_data

def test_load_knob_data():
    with TemporaryDirectory() as tmpdir:
        json_path = generate_knob_data(tmpdir, num_pairs=5)
        knob_types, volumes = load_knob_data(json_path)
        # The generator uses 5 unique knob types
        assert len(knob_types) == 5
        assert all(k in knob_types for k in ["gain", "bass", "mid", "treble", "presence"])
        assert len(volumes) == 5  # All volumes should be unique

if __name__ == "__main__":
    test_load_knob_data()
    print("test_load_knob_data passed.") 