import json as _json
import stat as _stat
import wave as _wave
from pathlib import Path as _Path

import pytest as _pytest

from nam.capture import al_runner as _al_runner
from nam.capture.export import AL_NY as _AL_NY
from nam.capture.export import build_al_data_config as _build_al_data_config
from nam.capture.export import build_al_model_config as _build_al_model_config
from nam.capture.params import KnobSpec as _KnobSpec
from nam.capture.project import CaptureProject as _CaptureProject
from nam.capture.project import PROJECT_FILENAME as _PROJECT_FILENAME
from nam.capture.project import QAModel as _QAModel
from nam.capture.project import mark_captured as _mark_captured
from nam.capture.project import new_project as _new_project
from nam.capture.project import save_project as _save_project


def _knobs() -> list[_KnobSpec]:
    return [
        _KnobSpec(name="Gain", min=0.0, max=10.0, step=0.5, avoid_zero=True),
        _KnobSpec(name="Tone", min=0.0, max=10.0, step=1.0),
    ]


def _project(**kwargs) -> _CaptureProject:
    defaults = dict(n_train=3, n_validation=2, seed=0)
    defaults.update(kwargs)
    return _new_project(_knobs(), **defaults)


def _write_wav(path: _Path, *, num_frames: int, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _wave.open(str(path), "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(rate)
        fp.writeframes(b"\x00\x00" * num_frames)


def _write_round(project_dir: _Path, round_idx: int, records: list[dict]) -> None:
    al_dir = _Path(project_dir) / _al_runner.AL_DIRNAME
    al_dir.mkdir(parents=True, exist_ok=True)
    (al_dir / f"proposed_captures_round_{round_idx}.json").write_text(_json.dumps(records))


def _prepared_project(tmp_path: _Path) -> _CaptureProject:
    """A project with a captured train and validation entry and real (silent) input WAVs."""
    project = _project(n_train=2, n_validation=1)
    _write_wav(tmp_path / project.train_input, num_frames=48_000 * 5, rate=48_000)
    _write_wav(tmp_path / project.validation_input, num_frames=48_000 * 5, rate=48_000)
    for entry in project.entries_for_split("train"):
        _mark_captured(entry, delay=0, qa=_QAModel())
    for entry in project.entries_for_split("validation"):
        _mark_captured(entry, delay=0, qa=_QAModel())
    _save_project(project, tmp_path)
    return project


def _fake_round_result(round_idx: int):
    from nam.train.active_learning import RoundResult

    return RoundResult(
        round_idx=round_idx,
        checkpoint_paths=[],
        candidates=[],
        selected=[],
        proposals=[
            {
                "params": {"Gain": 1.0, "Tone": 2.0},
                "score": 0.9,
                "y_path": f"captures/r{round_idx}_g1_t2.wav",
            }
        ],
        proposals_path=_Path(_al_runner.AL_DIRNAME) / f"proposed_captures_round_{round_idx}.json",
        aggregated_data_config={},
        aggregated_config_path=_Path(_al_runner.AL_DIRNAME) / f"aggregated_data_config_{round_idx}.json",
    )


# --- build_al_model_config -------------------------------------------------------


def test_build_al_model_config_shape_and_params_keep_step_and_avoid_zero():
    project = _project()

    config = _build_al_model_config(project)

    assert config["net"]["name"] == "ConcatLSTM"
    net_config = config["net"]["config"]
    params = net_config["params"]
    assert [p["name"] for p in params] == [knob.name for knob in project.knobs]
    for spec, knob in zip(params, project.knobs):
        assert spec["step"] == knob.step
        assert spec["avoid_zero"] == knob.avoid_zero
        assert "enum_names" not in spec
    assert net_config["hidden_size"] == 18
    assert net_config["num_layers"] == 3
    assert net_config["train_burn_in"] == 8192
    assert net_config["train_truncate"] is None


def test_build_al_model_config_scales_learning_rate_with_batch_size():
    from nam.capture.export import AL_REFERENCE_BATCH_SIZE as _ref_bs
    from nam.capture.export import AL_REFERENCE_LEARNING_RATE as _ref_lr

    project = _project()

    def lr(batch_size: int) -> float:
        return _build_al_model_config(project, batch_size=batch_size)["optimizer"]["lr"]

    # lr equals the reference at the reference batch, and scales as sqrt(batch) elsewhere.
    assert lr(_ref_bs) == _pytest.approx(_ref_lr)
    assert lr(4 * _ref_bs) == _pytest.approx(_ref_lr * 2.0)  # sqrt(4) = 2
    assert lr(_ref_bs // 4) == _pytest.approx(_ref_lr * 0.5)  # sqrt(1/4) = 1/2


def test_build_al_model_config_is_loadable_by_concat_lstm():
    from nam.models.parametric import ConcatLSTM

    project = _project()
    config = _build_al_model_config(project)

    net = ConcatLSTM.init_from_config(config["net"]["config"])

    assert net is not None
    assert net.param_dim == len(project.knobs)


# --- build_al_data_config ---------------------------------------------------------


def test_build_al_data_config_captured_only_and_ny_override_on_both_splits():
    project = _project()
    project.train_window.start_seconds = 1.0
    project.train_window.stop_seconds = 9.0
    project.validation_window.start_seconds = 2.0
    project.validation_window.stop_seconds = None

    train_entry = project.entries_for_split("train")[0]
    validation_entry = project.entries_for_split("validation")[0]
    _mark_captured(train_entry, delay=5, qa=_QAModel())
    _mark_captured(validation_entry, delay=2, qa=_QAModel())

    config = _build_al_data_config(project)

    assert len(config["train"]) == 1
    assert len(config["validation"]) == 1
    train_row = config["train"][0]
    validation_row = config["validation"][0]
    assert train_row["ny"] == _AL_NY
    assert validation_row["ny"] == _AL_NY
    assert train_row["start_seconds"] == 1.0
    assert train_row["stop_seconds"] == 9.0
    assert validation_row["start_seconds"] == 2.0
    assert validation_row["stop_seconds"] is None


# --- build_al_learning_config -----------------------------------------------------


def test_build_al_learning_config_val_batch_decoupled_and_defaults():
    from nam.capture.export import build_al_learning_config

    cfg = build_al_learning_config(batch_size=8, drop_last=True, val_batch_size=200)
    assert cfg["train_dataloader"]["batch_size"] == 8
    assert cfg["train_dataloader"]["drop_last"] is True
    assert cfg["val_dataloader"]["batch_size"] == 200

    # val_batch_size falls back to batch_size when unset.
    default_cfg = build_al_learning_config(batch_size=8, drop_last=True)
    assert default_cfg["val_dataloader"]["batch_size"] == 8


# --- compute_al_batch_size ---------------------------------------------------------


def test_compute_al_batch_size_targets_steps_per_epoch():
    # With ample memory, the batch is chosen so training does ~AL_TARGET_STEPS_PER_EPOCH
    # optimizer steps per epoch.
    target = _al_runner.AL_TARGET_STEPS_PER_EPOCH
    windows = 30 * target  # evenly divisible for a clean assertion
    batch, drop_last = _al_runner.compute_al_batch_size(10**15, 32768, windows)

    assert batch == round(windows / target)
    assert windows // batch == target
    assert drop_last is True


def test_compute_al_batch_size_grows_batch_with_dataset():
    # As the capture set grows over rounds, the batch grows but steps/epoch stays ~constant.
    target = _al_runner.AL_TARGET_STEPS_PER_EPOCH
    small, _ = _al_runner.compute_al_batch_size(10**15, 32768, 550)
    large, _ = _al_runner.compute_al_batch_size(10**15, 32768, 1650)

    assert small == round(550 / target)
    assert large == round(1650 / target)
    assert large > small
    assert abs(550 // small - target) <= 2
    assert abs(1650 // large - target) <= 2


def test_compute_al_batch_size_tiny_dataset_prefers_many_steps():
    # A tiny dataset is sized down toward batch 1 (maximal steps), not packed into one
    # big batch the way memory-driven sizing used to do.
    batch, drop_last = _al_runner.compute_al_batch_size(10**15, 32768, 7)
    assert batch == 1
    assert drop_last is True


def test_compute_al_batch_size_memory_is_only_a_ceiling():
    # A large step-target batch is clamped down to what memory allows; the memory cap
    # only ever lowers the batch, never raises it.
    bytes_per_window = 5120 * 32768
    available = 4 * bytes_per_window  # budgets 0.75*4 = 3 windows
    batch, _ = _al_runner.compute_al_batch_size(available, 32768, 10_000)
    assert batch == int(available * 0.75) // bytes_per_window == 3


def test_compute_al_batch_size_unknown_dataset_falls_back_to_memory_ceiling():
    bytes_per_window = 5120 * 32768
    available = 200 * bytes_per_window
    batch, drop_last = _al_runner.compute_al_batch_size(available, 32768, 0)

    assert batch == int(available * 0.75) // bytes_per_window
    assert drop_last is True


def test_compute_al_batch_size_clamps_to_256():
    batch, _ = _al_runner.compute_al_batch_size(10**15, 32768, 0)
    assert batch == 256


def test_compute_al_batch_size_tiny_memory_floors_at_one():
    batch, drop_last = _al_runner.compute_al_batch_size(1, 32768, 100)
    assert batch == 1
    assert drop_last is True


def test_compute_al_batch_size_drop_last_flips_to_false_past_10_percent_remainder(
    monkeypatch: _pytest.MonkeyPatch,
):
    # drop_last stays True in normal use; it flips off only if the step target ever makes
    # the batch large enough that the dropped tail exceeds 10% of the windows.
    monkeypatch.setattr(_al_runner, "AL_TARGET_STEPS_PER_EPOCH", 3)
    batch, drop_last = _al_runner.compute_al_batch_size(10**15, 32768, 20)
    assert batch == 7  # round(20 / 3)
    assert drop_last is False  # dropped = 20 % 7 = 6 > max(1, 20 // 10) = 2


def test_compute_al_batch_size_drop_last_stays_true_for_even_split():
    batch, drop_last = _al_runner.compute_al_batch_size(10**15, 32768, 900)
    assert batch == 30
    assert drop_last is True  # 900 % 30 == 0


# --- compute_al_val_batch_size -----------------------------------------------------


def test_compute_al_val_batch_size_is_memory_ceiling_ignoring_dataset():
    # Validation is sized by memory only (no steps-per-epoch concern, no dataset arg).
    assert _al_runner.compute_al_val_batch_size(10**15, 32768) == 256  # AL_MAX clamp
    bytes_per_window = 5120 * 32768
    available = 4 * bytes_per_window
    assert (
        _al_runner.compute_al_val_batch_size(available, 32768)
        == int(available * 0.75) // bytes_per_window
        == 3
    )
    assert _al_runner.compute_al_val_batch_size(1, 32768) == 1  # floor


def test_compute_al_val_batch_size_exceeds_step_sized_train_batch():
    # With ample memory the validation batch is much larger than the step-sized train batch,
    # so validation runs in as few passes as possible (a stable whole-set ESR).
    available = 10**15
    train_batch, _ = _al_runner.compute_al_batch_size(available, 32768, 550)
    assert _al_runner.compute_al_val_batch_size(available, 32768) > train_batch


# --- next_round_idx / load_round_proposals -----------------------------------------


def test_next_round_idx_with_no_rounds(tmp_path: _Path):
    assert _al_runner.next_round_idx(tmp_path) == 0


def test_next_round_idx_with_non_contiguous_rounds(tmp_path: _Path):
    _write_round(tmp_path, 0, [])
    _write_round(tmp_path, 2, [])
    assert _al_runner.next_round_idx(tmp_path) == 3


def test_next_round_idx_ignores_junk_filenames(tmp_path: _Path):
    _write_round(tmp_path, 1, [])
    al_dir = tmp_path / _al_runner.AL_DIRNAME
    (al_dir / "proposed_captures_round_abc.json").write_text("[]")
    (al_dir / "not_a_round_file.json").write_text("[]")
    assert _al_runner.next_round_idx(tmp_path) == 2


def test_load_round_proposals_parses_file(tmp_path: _Path):
    records = [{"params": {"Gain": 1.0}, "score": 0.5, "y_path": "captures/r0_g1.wav"}]
    _write_round(tmp_path, 0, records)
    assert _al_runner.load_round_proposals(tmp_path, 0) == records


# --- unimported_rounds / outstanding_proposal_y_paths / import_round_proposals -----


def test_unimported_rounds_detects_round_not_yet_imported(tmp_path: _Path):
    project = _project()
    _write_round(
        tmp_path,
        0,
        [{"params": {"Gain": 1.0, "Tone": 2.0}, "score": 0.9, "y_path": "captures/r0_a.wav"}],
    )
    assert _al_runner.unimported_rounds(project, tmp_path) == [0]


def test_unimported_rounds_empty_proposal_list_counts_as_imported(tmp_path: _Path):
    project = _project()
    _write_round(tmp_path, 0, [])
    assert _al_runner.unimported_rounds(project, tmp_path) == []


def test_import_round_proposals_appends_pending_train_entries_with_continuing_indices():
    project = _project(n_train=2, n_validation=1)
    max_train_index = max(entry.index for entry in project.entries_for_split("train"))
    records = [
        {"params": {"Gain": 1.0, "Tone": 2.0}, "score": 0.9, "y_path": "captures/r0_a.wav"},
        {"params": {"Gain": 3.0, "Tone": 4.0}, "score": 0.8, "y_path": "captures/r0_b.wav"},
    ]

    appended = _al_runner.import_round_proposals(project, records)

    assert [entry.y_path for entry in appended] == ["captures/r0_a.wav", "captures/r0_b.wav"]
    assert all(entry.split == "train" for entry in appended)
    assert all(entry.status == "pending" for entry in appended)
    assert [entry.index for entry in appended] == [max_train_index + 1, max_train_index + 2]
    assert appended[0] in project.entries
    assert len(project.entries_for_split("train")) == 2 + 2


def test_import_round_proposals_second_import_is_a_noop():
    project = _project(n_train=2, n_validation=1)
    records = [
        {"params": {"Gain": 1.0, "Tone": 2.0}, "score": 0.9, "y_path": "captures/r0_a.wav"},
    ]

    first = _al_runner.import_round_proposals(project, records)
    second = _al_runner.import_round_proposals(project, records)

    assert len(first) == 1
    assert second == []
    assert len(project.entries_for_split("train")) == 2 + 1


def test_outstanding_proposal_y_paths_pending_then_captured(tmp_path: _Path):
    project = _project(n_train=2, n_validation=1)
    records = [
        {"params": {"Gain": 1.0, "Tone": 2.0}, "score": 0.9, "y_path": "captures/r0_a.wav"},
        {"params": {"Gain": 3.0, "Tone": 4.0}, "score": 0.8, "y_path": "captures/r0_b.wav"},
    ]
    _write_round(tmp_path, 0, records)

    # Not imported at all yet: outstanding.
    outstanding = _al_runner.outstanding_proposal_y_paths(project, tmp_path)
    assert set(outstanding) == {"captures/r0_a.wav", "captures/r0_b.wav"}

    # Imported but still pending: still outstanding.
    _al_runner.import_round_proposals(project, records)
    outstanding = _al_runner.outstanding_proposal_y_paths(project, tmp_path)
    assert set(outstanding) == {"captures/r0_a.wav", "captures/r0_b.wav"}

    # Captured: no longer outstanding.
    for entry in project.entries_for_split("train"):
        if entry.y_path in {"captures/r0_a.wav", "captures/r0_b.wav"}:
            _mark_captured(entry, delay=0, qa=_QAModel())
    assert _al_runner.outstanding_proposal_y_paths(project, tmp_path) == []


# --- run_active_learning_round -------------------------------------------------------


def test_run_active_learning_round_refuses_without_captured_train_entries(tmp_path: _Path):
    project = _project(n_train=2, n_validation=1)
    _save_project(project, tmp_path)

    with _pytest.raises(RuntimeError):
        _al_runner.run_active_learning_round(tmp_path)


def test_run_active_learning_round_refuses_without_captured_validation_entries(tmp_path: _Path):
    project = _project(n_train=2, n_validation=1)
    for entry in project.entries_for_split("train"):
        _mark_captured(entry, delay=0, qa=_QAModel())
    _save_project(project, tmp_path)

    with _pytest.raises(RuntimeError):
        _al_runner.run_active_learning_round(tmp_path)


def test_run_active_learning_round_refuses_with_outstanding_proposals_unless_forced(
    tmp_path: _Path, monkeypatch: _pytest.MonkeyPatch
):
    _prepared_project(tmp_path)
    _write_round(
        tmp_path,
        0,
        [{"params": {"Gain": 1.0, "Tone": 2.0}, "score": 0.9, "y_path": "captures/r0_pending.wav"}],
    )

    with _pytest.raises(RuntimeError):
        _al_runner.run_active_learning_round(tmp_path)

    monkeypatch.setattr(
        "nam.train.active_learning.run_round",
        lambda **kwargs: _fake_round_result(kwargs["round_idx"]),
    )

    result = _al_runner.run_active_learning_round(tmp_path, force=True)
    assert result.round_idx == 1


def test_run_active_learning_round_writes_configs_and_delegates_to_run_round(
    tmp_path: _Path, monkeypatch: _pytest.MonkeyPatch
):
    project = _prepared_project(tmp_path)
    project_json_before = (tmp_path / _PROJECT_FILENAME).read_text()

    captured_kwargs: dict = {}

    def fake_run_round(**kwargs):
        captured_kwargs.update(kwargs)
        assert _Path.cwd().samefile(tmp_path)
        return _fake_round_result(kwargs["round_idx"])

    monkeypatch.setattr("nam.train.active_learning.run_round", fake_run_round)

    prior_cwd = _Path.cwd()
    result = _al_runner.run_active_learning_round(
        tmp_path, ensemble_size=1, num_restarts=1, num_steps=1
    )
    assert _Path.cwd().samefile(prior_cwd)

    assert result.round_idx == 0
    assert captured_kwargs["round_idx"] == 0
    assert captured_kwargs["output_dir"] == _Path(_al_runner.AL_DIRNAME)
    assert captured_kwargs["y_path_prefix"] == "captures/r"
    assert captured_kwargs["g_opt_input_wav"] == project.train_input
    assert captured_kwargs["data_config"]["type"] == "parametric"
    assert captured_kwargs["model_config"]["net"]["name"] == "ConcatLSTM"
    learning_config = captured_kwargs["learning_config"]
    assert "batch_size" in learning_config["train_dataloader"]
    # Validation batch is sized independently (memory ceiling) and is never smaller than
    # the step-sized training batch.
    assert (
        learning_config["val_dataloader"]["batch_size"]
        >= learning_config["train_dataloader"]["batch_size"]
    )

    al_dir = tmp_path / _al_runner.AL_DIRNAME
    assert (al_dir / "model.json").is_file()
    assert (al_dir / "learning.json").is_file()
    assert (al_dir / "data.json").is_file()

    script_path = tmp_path / _al_runner.RUNNER_SCRIPT_FILENAME
    assert script_path.is_file()
    assert script_path.stat().st_mode & _stat.S_IXUSR

    assert (tmp_path / _PROJECT_FILENAME).read_text() == project_json_before


def test_run_active_learning_round_explicit_batch_size_forces_drop_last_true(
    tmp_path: _Path, monkeypatch: _pytest.MonkeyPatch
):
    _prepared_project(tmp_path)

    captured_kwargs: dict = {}
    monkeypatch.setattr(
        "nam.train.active_learning.run_round",
        lambda **kwargs: captured_kwargs.update(kwargs) or _fake_round_result(kwargs["round_idx"]),
    )

    _al_runner.run_active_learning_round(tmp_path, batch_size=3)

    al_dir = tmp_path / _al_runner.AL_DIRNAME
    with (al_dir / "learning.json").open() as fp:
        learning_config = _json.load(fp)
    assert learning_config["train_dataloader"]["batch_size"] == 3
    assert learning_config["train_dataloader"]["drop_last"] is True
    assert learning_config["val_dataloader"]["batch_size"] == 3
    with (al_dir / "model.json").open() as fp:
        model_config = _json.load(fp)
    from nam.capture.export import active_learning_learning_rate as _al_lr

    assert model_config["optimizer"]["lr"] == _pytest.approx(_al_lr(3))


@_pytest.mark.parametrize("batch_size", [0, 257])
def test_run_active_learning_round_rejects_batch_size_above_limit(
    tmp_path: _Path, batch_size: int
):
    _prepared_project(tmp_path)

    with _pytest.raises(ValueError, match="batch_size must be in"):
        _al_runner.run_active_learning_round(tmp_path, batch_size=batch_size)
