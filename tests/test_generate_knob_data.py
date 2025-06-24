import os
import json
from tempfile import TemporaryDirectory
from nam.generate_knob_data import generate_knob_data

def test_generate_knob_data():
    with TemporaryDirectory() as tmpdir:
        json_path = generate_knob_data(tmpdir, num_pairs=5)
        assert os.path.exists(json_path)
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        assert len(dataset) == 5
        for i, entry in enumerate(dataset):
            assert os.path.exists(entry['input_path'])
            assert os.path.exists(entry['output_path'])
            assert 'knob_type' in entry
            assert 'knob_level' in entry
            assert 'volume' in entry

if __name__ == "__main__":
    test_generate_knob_data()
    print("test_generate_knob_data passed.") 