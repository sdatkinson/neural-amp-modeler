from nam.train import colab as _colab


def test_colab_run_does_not_forward_removed_core_kwargs(monkeypatch):
    monkeypatch.setattr(_colab, "_check_for_files", lambda: "input.wav")
    captured_kwargs = {}

    def fake_train(**kwargs):
        captured_kwargs.update(kwargs)
        return _colab._TrainOutput(model=None, metadata=None)

    monkeypatch.setattr(_colab, "_train", fake_train)

    _colab.run(epochs=1, delay=2, seed=None, ignore_checks=True)

    removed_kwargs = {"model_type", "architecture", "lr", "lr_decay", "fit_mrstft"}
    forwarded = removed_kwargs & captured_kwargs.keys()
    assert not forwarded
