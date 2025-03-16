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

from nam.models import exportable, metadata
from nam.train import metadata as train_metadata


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
        "user_metadata,other_metadata",
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
                    input_level_dbu=-6.5,
                    output_level_dbu=-12.5,
                ),
                None,
            ),
            (
                None,
                {
                    train_metadata.TRAINING_KEY: train_metadata.TrainingMetadata(
                        settings=train_metadata.Settings(
                            fit_cab=True, ignore_checks=False
                        ),
                        data=train_metadata.Data(
                            latency=train_metadata.Latency(
                                manual=None,
                                calibration=train_metadata.LatencyCalibration(
                                    algorithm_version=1,
                                    delays=[1, 3],
                                    safety_factor=4,
                                    recommended=-3,
                                    warnings=train_metadata.LatencyCalibrationWarnings(
                                        matches_lookahead=False,
                                        disagreement_too_high=False,
                                    ),
                                ),
                            ),
                            checks=train_metadata.DataChecks(version=4, passed=True),
                        ),
                        validation_esr=0.01,
                    ).model_dump()
                },
            ),
            (
                metadata.UserMetadata(
                    name="My Model",
                    modeled_by="Steve",
                    gear_type=metadata.GearType.AMP,
                    gear_make="SteveCo",
                    gear_model="SteveAmp",
                    tone_type=metadata.ToneType.HI_GAIN,
                ),
                {
                    train_metadata.TRAINING_KEY: train_metadata.TrainingMetadata(
                        settings=train_metadata.Settings(
                            fit_cab=True, ignore_checks=False
                        ),
                        data=train_metadata.Data(
                            latency=train_metadata.Latency(
                                manual=None,
                                calibration=train_metadata.LatencyCalibration(
                                    algorithm_version=1,
                                    delays=[1, 3],
                                    safety_factor=4,
                                    recommended=-3,
                                    warnings=train_metadata.LatencyCalibrationWarnings(
                                        matches_lookahead=False,
                                        disagreement_too_high=False,
                                    ),
                                ),
                            ),
                            checks=train_metadata.DataChecks(version=4, passed=True),
                        ),
                        validation_esr=0.01,
                    ).model_dump()
                },
            ),
        ),
    )
    def test_export_metadata(
        self,
        user_metadata: Optional[metadata.UserMetadata],
        other_metadata: Optional[dict],
    ):
        """
        Assert export behavior when metadata is provided
        """

        def assert_metadata(actual: dict, expected: dict):
            assert isinstance(actual, dict)
            for key, expected_value in expected.items():
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
                tmpdir, user_metadata=user_metadata, other_metadata=other_metadata
            )
            model_basename = "model.nam"
            model_path = Path(tmpdir, model_basename)
            assert model_path.exists()
            with open(model_path, "r") as fp:
                model_dict = json.load(fp)
            metadata_key = "metadata"
            training_key = train_metadata.TRAINING_KEY
            assert metadata_key in model_dict
            model_dict_metadata = model_dict[metadata_key]
            if user_metadata is not None:
                assert_metadata(model_dict_metadata, user_metadata.model_dump())
            if other_metadata is not None:
                assert training_key in model_dict_metadata
                assert_metadata(model_dict_metadata, other_metadata)

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
