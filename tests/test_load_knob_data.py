import os
from tempfile import TemporaryDirectory
from nam.generate_knob_data import generate_knob_data
from nam.load_knob_data import load_knob_data

def test_load_knob_data():
    with TemporaryDirectory() as tmpdir:
        json_path = generate_knob_data(tmpdir, num_pairs=5)
        dataset = load_knob_data(json_path)
        # Extract unique knob types and levels from dataset
        knob_types = {entry['knob_type'] for entry in dataset}
        knob_levels = {entry['knob_level'] for entry in dataset}
        expected_knob_types = ["volume", "gain", "bass", "mid", "treble"]
        assert len(knob_types) == 5
        assert all(k in knob_types for k in expected_knob_types)
        # Only allow knob levels 0.1, 0.2, ..., 1.0, but for 5 pairs: 0.1, 0.2, 0.3, 0.4, 0.5
        expected_knob_levels = [round((i % 10 + 1) / 10, 1) for i in range(5)]
        assert knob_levels == set(expected_knob_levels)
        # Check dataset structure
        assert isinstance(dataset, list)
        assert len(dataset) == 5
        for entry in dataset:
            assert 'input_path' in entry
            assert 'output_path' in entry
            assert 'knob_type' in entry
            assert 'knob_level' in entry
            assert os.path.exists(entry['input_path'])
            assert os.path.exists(entry['output_path'])

if __name__ == "__main__":
    test_load_knob_data()
    print("test_load_knob_data passed.") 