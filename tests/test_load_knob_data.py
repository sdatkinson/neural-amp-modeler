import os
from tempfile import TemporaryDirectory
from nam.generate_knob_data import generate_knob_data
from nam.load_knob_data import load_knob_data

def test_load_knob_data():
    with TemporaryDirectory() as tmpdir:
        json_path = generate_knob_data(tmpdir, num_pairs=5)
        knob_types, knob_levels = load_knob_data(json_path)
        # The generator uses 6 unique knob types, but only 5 pairs, so only first 5 are used
        expected_knob_types = ["volume", "gain", "bass", "mid", "treble"]
        assert len(knob_types) == 5
        assert all(k in knob_types for k in expected_knob_types)
        # Only allow knob levels 0.1, 0.2, ..., 1.0, but for 5 pairs: 0.1, 0.2, 0.3, 0.4, 0.5
        expected_knob_levels = [round((i % 10 + 1) / 10, 1) for i in range(5)]
        assert knob_levels == set(expected_knob_levels)

if __name__ == "__main__":
    test_load_knob_data()
    print("test_load_knob_data passed.") 