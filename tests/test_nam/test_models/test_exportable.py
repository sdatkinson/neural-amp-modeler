# File: test_exportable.py
# Created Date: Sunday January 29th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Test export behavior of models
"""

import json
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from pydantic import BaseModel

from nam.models import exportable
from nam.models import metadata


class TestExportable(object):
    def test_export(self):
        """
        Does it work?
        """

        model = self._get_model()
        with TemporaryDirectory() as tmpdir:
            model.export(tmpdir)
            model_basename = "model.nam"
            model_path = Path(tmpdir, model_basename)
            assert model_path.exists()
            with open(model_path, "r") as fp:
                model_dict = json.load(fp)
            required_keys = {"version", "architecture", "config", "weights"}
            for key in required_keys:
                assert key in model_dict
            weights_list = model_dict["weights"]
            assert isinstance(weights_list, list)
            assert len(weights_list) == 2
            assert all(isinstance(w, float) for w in weights_list)

    @pytest.mark.parametrize(
        "user_metadata,training_metadata",
        (
            (None, None),
            (metadata.UserMetadata(), None),
            (
                metadata.UserMetadata(
                    name="My Model",
                    modeled_by="Steve",
                    gear_type=metadata.GearType.AMP,
                    gear_make="SteveCo",
                    gear_model="SteveAmp",
                    tone_type=metadata.ToneType.HI_GAIN,
                ),
                None,
            ),
            (
                None,
                metadata.TrainingMetadata(
                    validation_esr=0.01,
                    fit_cab=True,
                    ignored_checks=False,
                    latency=metadata.LatencyMetadata(
                        user_samples=None,
                        estimation_algorithm_version=1,
                        estimated_samples=[1, -2],
                        safety_factor_samples=4,
                    ),
                ),
            ),
        ),
    )
    def test_export_metadata(
        self,
        user_metadata: Optional[metadata.UserMetadata],
        training_metadata: Optional[metadata.TrainingMetadata],
    ):
        """
        Assert export behavior when metadata is provided
        """

        def assert_metadata(actual: dict, expected: BaseModel):
            assert isinstance(actual, dict)
            for key, expected_value in expected.model_dump().items():
                assert key in actual
                actual_value = actual[key]
                if isinstance(expected_value, BaseModel):
                    assert_metadata(actual_value, expected_value)
                else:
                    if isinstance(expected_value, Enum):
                        expected_value = expected_value.value
                    assert actual_value == expected_value

        model = self._get_model()
        with TemporaryDirectory() as tmpdir:
            model.export(
                tmpdir, user_metadata=user_metadata, training_metadata=training_metadata
            )
            model_basename = "model.nam"
            model_path = Path(tmpdir, model_basename)
            assert model_path.exists()
            with open(model_path, "r") as fp:
                model_dict = json.load(fp)
            metadata_key = "metadata"
            training_key = "training"
            assert metadata_key in model_dict
            model_dict_metadata = model_dict[metadata_key]
            if user_metadata is not None:
                assert_metadata(model_dict_metadata, user_metadata)
            if training_metadata is not None:
                assert training_key in model_dict_metadata
                actual_training_metadata = model_dict_metadata[training_key]
                assert_metadata(actual_training_metadata, training_metadata)

    @pytest.mark.parametrize("include_snapshot", (True, False))
    def test_include_snapshot(self, include_snapshot):
        """
        Does the option to include a snapshot work?
        """
        model = self._get_model()

        with TemporaryDirectory() as tmpdir:
            model.export(tmpdir, include_snapshot=include_snapshot)
            input_path = Path(tmpdir, "test_inputs.npy")
            output_path = Path(tmpdir, "test_outputs.npy")
            if include_snapshot:
                assert input_path.exists()
                assert output_path.exists()
                # And check that the output is correct
                x = np.load(input_path)
                y = np.load(output_path)
                preds = model(torch.Tensor(x)).detach().cpu().numpy()
                assert preds == pytest.approx(y)
            else:
                assert not input_path.exists()
                assert not output_path.exists()

    @classmethod
    def _get_model(cls):
        class Model(nn.Module, exportable.Exportable):
            def __init__(self):
                super().__init__()
                self._scale = nn.Parameter(torch.tensor(0.0))
                self._bias = nn.Parameter(torch.tensor(0.0))

            def forward(self, x: torch.Tensor):
                return self._scale * x + self._bias

            def export_cpp_header(self, filename: Path):
                pass

            def _export_config(self):
                return {}

            def _export_input_output(self) -> Tuple[np.ndarray, np.ndarray]:
                x = 0.01 * np.random.randn(
                    3,
                )
                y = self(torch.Tensor(x)).detach().cpu().numpy()
                return x, y

            def _export_weights(self) -> np.ndarray:
                return torch.stack([self._scale, self._bias]).detach().cpu().numpy()

        return Model()
