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
    assert config["loss"] == {
        "val_loss": "esr",
        "mel_weight": _pytest.approx(6.2e-05),
    }
    assert "gradient_reject_norm" not in config["optimizer"]


def test_build_al_model_config_uses_reference_learning_rate_for_all_physical_batches():
    from nam.capture.export import AL_REFERENCE_BATCH_SIZE as _ref_bs
    from nam.capture.export import AL_REFERENCE_LEARNING_RATE as _ref_lr

    project = _project()

    def lr(batch_size: int) -> float:
        return _build_al_model_config(project, batch_size=batch_size)["optimizer"]["lr"]

    assert _ref_bs == 37
    assert _ref_lr == _pytest.approx(0.008)
    assert lr(_ref_bs) == _pytest.approx(_ref_lr)
    assert lr(32) == _pytest.approx(_ref_lr)


def test_active_learning_max_epochs_is_fixed():
    from nam.capture.export import AL_DEFAULT_MAX_EPOCHS as _default
    from nam.capture.export import active_learning_max_epochs as _epochs

    assert _default == 50
    assert _epochs(1) == _default
    assert _epochs(600) == _default
    assert _epochs(0) == _default


def test_active_learning_steps_per_epoch_respects_drop_last():
    from nam.capture.export import active_learning_steps_per_epoch as _steps

    assert _steps(100, 30, drop_last=True) == 3  # floor: final partial batch dropped
    assert _steps(100, 30, drop_last=False) == 4  # ceil: final partial batch kept
    assert _steps(0, 30, drop_last=True) == 0  # unknown window count


def test_active_learning_lr_gamma_matches_panama():
    from nam.capture.export import active_learning_lr_gamma as _gamma

    assert _gamma(50) == _pytest.approx(0.995)
    assert _gamma(600) == _pytest.approx(0.995)


def test_build_al_model_config_uses_fixed_panama_gamma():
    from nam.capture.export import active_learning_lr_gamma as _gamma

    project = _project()

    def kwargs(max_epochs: int) -> dict:
        cfg = _build_al_model_config(project, max_epochs=max_epochs)
        return cfg["lr_scheduler"]["kwargs"]

    assert kwargs(50)["gamma"] == _pytest.approx(_gamma(50))
    assert kwargs(30)["gamma"] == _pytest.approx(_gamma(30))
    assert kwargs(30)["gamma"] == _pytest.approx(0.995)


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
    assert cfg["trainer"]["accumulate_grad_batches"] == 1
    assert cfg["batch_sizing"]["target_effective_batch_size"] == 37

    # val_batch_size falls back to batch_size when unset.
    default_cfg = build_al_learning_config(batch_size=8, drop_last=True)
    assert default_cfg["val_dataloader"]["batch_size"] == 8

    # max_epochs defaults to the reference schedule and is otherwise passed through.
    from nam.capture.export import AL_DEFAULT_MAX_EPOCHS as _default_epochs

    assert default_cfg["trainer"]["max_epochs"] == _default_epochs
    assert default_cfg["trainer"]["deterministic"] == "warn"
    assert default_cfg["trainer"]["benchmark"] is False
    assert (
        build_al_learning_config(batch_size=8, drop_last=True, max_epochs=42)["trainer"][
            "max_epochs"
        ]
        == 42
    )

    # The standard 1.0 gradient clip: a tighter clip distorts Adam's second-moment estimate
    # during a spike and destabilizes training, so the clip stays at the usual 1.0.
    from nam.capture.export import AL_GRADIENT_CLIP_VAL as _clip

    assert default_cfg["trainer"]["gradient_clip_val"] == _clip
    assert _clip == 1.0


# --- compute_al_batch_size ---------------------------------------------------------


def test_compute_al_batch_plan_uses_target_batch_when_it_fits():
    physical, accumulation, effective = _al_runner.compute_al_batch_plan(
        10**15, _AL_NY, 2_000
    )

    assert (physical, accumulation, effective) == (37, 1, 37)


def test_compute_al_batch_plan_accumulates_on_smaller_gpu():
    bytes_per_window = 5120 * _AL_NY
    available = 4 * bytes_per_window
    physical, accumulation, effective = _al_runner.compute_al_batch_plan(
        available, _AL_NY, 2_000
    )

    assert physical == 3
    assert accumulation == 12
    assert effective == 36


def test_compute_al_batch_plan_caps_target_at_tiny_dataset():
    physical, accumulation, effective = _al_runner.compute_al_batch_plan(
        10**15, _AL_NY, 7
    )

    assert (physical, accumulation, effective) == (7, 1, 7)


def test_compute_al_batch_size_memory_is_only_a_ceiling():
    # A large step-target batch is clamped down to what memory allows; the memory cap
    # only ever lowers the batch, never raises it.
    bytes_per_window = 5120 * _AL_NY
    available = 4 * bytes_per_window  # budgets 0.75*4 = 3 windows
    batch, _ = _al_runner.compute_al_batch_size(available, _AL_NY, 10_000)
    assert batch == int(available * 0.75) // bytes_per_window == 3


def test_compute_al_batch_size_unknown_dataset_uses_best_effective_batch_plan():
    bytes_per_window = 5120 * _AL_NY
    available = 200 * bytes_per_window
    batch, drop_last = _al_runner.compute_al_batch_size(available, _AL_NY, 0)

    assert batch == 37
    assert drop_last is True


def test_compute_al_batch_size_bounded_by_target_effective_batch():
    batch, _ = _al_runner.compute_al_batch_size(10**15, _AL_NY, 0)
    assert batch == 37


def test_compute_al_batch_size_tiny_memory_floors_at_one():
    batch, drop_last = _al_runner.compute_al_batch_size(1, _AL_NY, 100)
    assert batch == 1
    assert drop_last is True


# --- compute_al_val_batch_size -----------------------------------------------------


def test_compute_al_val_batch_size_is_memory_ceiling_ignoring_dataset():
    # Validation is sized by memory only (no steps-per-epoch concern, no dataset arg).
    assert _al_runner.compute_al_val_batch_size(10**15, _AL_NY) == 512
    bytes_per_window = 5120 * _AL_NY
    available = 4 * bytes_per_window
    assert (
        _al_runner.compute_al_val_batch_size(available, _AL_NY)
        == int(available * 0.75) // bytes_per_window
        == 3
    )
    assert _al_runner.compute_al_val_batch_size(1, _AL_NY) == 1


def test_compute_al_val_batch_size_at_least_train_batch_when_memory_allows():
    # Validation is memory-bounded (fewest passes); training is bounded by the smaller
    # effective-batch target. With ample memory the validation batch is therefore at least
    # as large as the training batch, and fills the memory ceiling.
    available = 10**15
    train_batch, _ = _al_runner.compute_al_batch_size(available, _AL_NY, 550)
    val_batch = _al_runner.compute_al_val_batch_size(available, _AL_NY)
    assert val_batch >= train_batch
    assert val_batch == 512


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
    assert learning_config["trainer"]["accumulate_grad_batches"] > 1
    assert learning_config["batch_sizing"]["auto"] is False
    with (al_dir / "model.json").open() as fp:
        model_config = _json.load(fp)
    from nam.capture.export import active_learning_learning_rate as _al_lr

    assert model_config["optimizer"]["lr"] == _pytest.approx(_al_lr(3))


@_pytest.mark.parametrize("batch_size", [0, 513])
def test_run_active_learning_round_rejects_batch_size_above_limit(
    tmp_path: _Path, batch_size: int
):
    _prepared_project(tmp_path)

    with _pytest.raises(ValueError, match="batch_size must be in"):
        _al_runner.run_active_learning_round(tmp_path, batch_size=batch_size)
