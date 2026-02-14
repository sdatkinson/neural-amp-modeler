# File: test_graceful_shutdown.py
# Created Date: Friday January 30th 2026
# Purpose: Test that graceful shutdown (Ctrl+C) still generates a model
#
# This script can be run standalone to test the graceful shutdown behavior,
# or used as a pytest test.

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Optional, Tuple

import numpy as np

# Add the parent directory to the path so we can import nam
sys.path.insert(0, str(Path(__file__).parent.parent))

from nam.data import np_to_wav


def _os_process_helpers() -> Dict[str, Callable]:
    """
    Return OS-appropriate implementations for process signaling and I/O.

    Detects the OS and returns a dict of functions with consistent signatures
    whose implementations differ by platform (Unix vs Windows).

    :return: Dict with keys:
        - get_popen_kwargs: () -> dict, extra kwargs for Popen (process group)
        - send_interrupt: (process: subprocess.Popen) -> None
        - kill_process_group: (process: subprocess.Popen) -> None
        - read_available_output: (process, timeout) -> str or None
    """
    if os.name == "nt":
        # Windows: no killpg/getpgid/setsid; use terminate and CREATE_NEW_PROCESS_GROUP

        def get_popen_kwargs() -> dict:
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

        def send_interrupt(process: subprocess.Popen) -> None:
            try:
                process.send_signal(signal.CTRL_C_EVENT)
            except (ProcessLookupError, ValueError):
                process.terminate()

        def kill_process_group(process: subprocess.Popen) -> None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except ProcessLookupError:
                pass

        def read_available_output(
            process: subprocess.Popen, timeout: float
        ) -> Optional[str]:
            # select() does not work with pipes on Windows; use thread-based polling
            import threading

            result: list = []

            def read():
                try:
                    line = process.stdout.readline()
                    if line:
                        result.append(line)
                except (ValueError, OSError):
                    pass

            t = threading.Thread(target=read)
            t.daemon = True
            t.start()
            t.join(timeout=timeout)
            return result[0] if result else None

    else:
        # Unix: use process groups, killpg, setsid

        def get_popen_kwargs() -> dict:
            return {"preexec_fn": os.setsid}

        def send_interrupt(process: subprocess.Popen) -> None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            except ProcessLookupError:
                pass

        def kill_process_group(process: subprocess.Popen) -> None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

        def read_available_output(
            process: subprocess.Popen, timeout: float
        ) -> Optional[str]:
            import select

            if select.select([process.stdout], [], [], timeout)[0]:
                return process.stdout.readline()
            return None

    return {
        "get_popen_kwargs": get_popen_kwargs,
        "send_interrupt": send_interrupt,
        "kill_process_group": kill_process_group,
        "read_available_output": read_available_output,
    }


def create_test_data(root_path: Path) -> Tuple[Path, Path]:
    """
    Create minimal test audio files for training.

    :return: (x_path, y_path) paths to input and output wav files
    """
    # Create enough samples for training and validation
    # We need enough for at least one epoch to complete and trigger a checkpoint
    num_samples = 2048  # Larger to allow for proper train/val split
    x = (np.random.rand(num_samples) - 0.5).astype(np.float32)
    y = (1.1 * x).astype(np.float32)  # Simple linear relationship

    input_dir = Path(root_path, "inputs")
    input_dir.mkdir(exist_ok=True)

    x_path = Path(input_dir, "input.wav")
    y_path = Path(input_dir, "output.wav")

    np_to_wav(x, x_path)
    np_to_wav(y, y_path)

    return x_path, y_path


def create_configs(
    root_path: Path, x_path: Path, y_path: Path, num_epochs: int = 100
) -> Tuple[Path, Path, Path]:
    """
    Create minimal config files for nam-full training.

    :return: (data_config_path, model_config_path, learning_config_path)
    """
    input_dir = Path(root_path, "inputs")

    # Data config - use simple sample-based splits
    # Use enough validation samples but leave room for training
    num_samples_validation = 256
    data_config = {
        "train": {
            "start": None,
            "stop": -num_samples_validation,
            "ny": 32,  # Output size per sample
        },
        "validation": {
            "start": -num_samples_validation,
            "stop": None,
            "ny": None,
        },
        "common": {
            "x_path": str(x_path),
            "y_path": str(y_path),
            "delay": 0,
            "require_input_pre_silence": None,
        },
    }

    # Minimal WaveNet model config
    model_config = {
        "net": {
            "name": "WaveNet",
            "config": {
                "layers_configs": [
                    {
                        "condition_size": 1,
                        "input_size": 1,
                        "channels": 2,
                        "head_size": 1,
                        "kernel_size": 3,
                        "dilations": [1],
                        "activation": "Tanh",
                        "head_bias": False,
                    },
                ],
                "head_scale": 0.02,
            },
        },
        "optimizer": {"lr": 0.004},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.993}},
    }

    # Learning config - CPU only, many epochs (we'll interrupt)
    learning_config = {
        "train_dataloader": {
            "batch_size": 2,
            "shuffle": True,
            "pin_memory": False,
            "drop_last": True,
            "num_workers": 0,
        },
        "val_dataloader": {},
        "trainer": {
            "max_epochs": num_epochs,
            # No GPU - use CPU for testing
        },
        "trainer_fit_kwargs": {},
    }

    # Write configs
    data_config_path = Path(input_dir, "data_config.json")
    model_config_path = Path(input_dir, "model_config.json")
    learning_config_path = Path(input_dir, "learning_config.json")

    with open(data_config_path, "w") as fp:
        json.dump(data_config, fp)
    with open(model_config_path, "w") as fp:
        json.dump(model_config, fp)
    with open(learning_config_path, "w") as fp:
        json.dump(learning_config, fp)

    return data_config_path, model_config_path, learning_config_path


def find_nam_files(directory: Path) -> list:
    """Find all .nam files in a directory tree."""
    return list(directory.rglob("*.nam"))


def run_training_with_interrupt(
    data_config_path: Path,
    model_config_path: Path,
    learning_config_path: Path,
    output_path: Path,
    interrupt_after_training_starts: bool = True,
    interrupt_delay: float = 2.0,
    timeout: float = 120.0,
    conda_env: Optional[str] = None,
) -> Tuple[int, bool, list]:
    """
    Run nam-full training and send SIGINT after training starts.

    :param interrupt_after_training_starts: If True, wait for training to start
        before sending interrupt. If False, use interrupt_delay from process start.
    :param interrupt_delay: Seconds to wait after training starts (or after
        process start if interrupt_after_training_starts is False)
    :param timeout: Maximum time to wait for process to complete
    :param conda_env: Name of conda environment to use
    :return: (exit_code, graceful_shutdown_detected, nam_files_found)
    """
    # Build command
    if conda_env:
        # Use conda run to execute in the environment
        cmd = [
            "conda",
            "run",
            "-n",
            conda_env,
            "--no-capture-output",
            "nam-full",
            str(data_config_path),
            str(model_config_path),
            str(learning_config_path),
            str(output_path),
            "--no-show",
            "--no-plots",
        ]
    else:
        cmd = [
            "nam-full",
            str(data_config_path),
            str(model_config_path),
            str(learning_config_path),
            str(output_path),
            "--no-show",
            "--no-plots",
        ]

    print(f"Starting training with command: {' '.join(cmd)}")

    helpers = _os_process_helpers()
    popen_kwargs = helpers["get_popen_kwargs"]()

    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **popen_kwargs,
    )

    output_lines = []
    graceful_shutdown_detected = False
    training_started = False
    training_start_time = None
    interrupt_sent = False

    start_time = time.time()

    try:
        # Wait for training to start, then send interrupt
        while True:
            # Check if process has ended
            if process.poll() is not None:
                break

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"Timeout after {timeout}s, killing process")
                helpers["kill_process_group"](process)
                break

            # Try to read output (non-blocking)
            try:
                line = helpers["read_available_output"](process, 0.1)
                if line:
                    output_lines.append(line)
                    print(f"[nam-full] {line.rstrip()}")

                    # Check for training indicators - these appear when training actually starts
                    # Look for PyTorch Lightning training indicators
                    training_indicators = [
                        "Epoch ",  # Lightning progress
                        "GPU available:",  # Lightning startup
                        "Trainer already configured",  # Lightning
                        "LOCAL_RANK:",  # Lightning distributed
                        "kick ass",  # Our message in core.py
                        "Sanity Checking",  # Lightning validation check
                    ]
                    if any(ind in line for ind in training_indicators):
                        if not training_started:
                            training_started = True
                            training_start_time = time.time()
                            print(f"\n>>> Training detected after {elapsed:.1f}s <<<\n")

                    if "graceful" in line.lower():
                        graceful_shutdown_detected = True
            except Exception:
                time.sleep(0.1)

            # Determine when to send interrupt
            should_send_interrupt = False
            if not interrupt_sent:
                if interrupt_after_training_starts:
                    # Wait for training to start, then wait interrupt_delay
                    if training_started and training_start_time is not None:
                        time_since_training = time.time() - training_start_time
                        if time_since_training >= interrupt_delay:
                            should_send_interrupt = True
                else:
                    # Just use delay from process start
                    if elapsed >= interrupt_delay:
                        should_send_interrupt = True

            if should_send_interrupt and process.poll() is None:
                print(f"\n>>> Sending SIGINT (elapsed={elapsed:.1f}s) <<<\n")
                helpers["send_interrupt"](process)
                interrupt_sent = True

        # Read any remaining output
        remaining_output, _ = process.communicate(timeout=30)
        if remaining_output:
            for line in remaining_output.split("\n"):
                if line:
                    output_lines.append(line)
                    print(f"[nam-full] {line}")
                    if "graceful" in line.lower():
                        graceful_shutdown_detected = True

    except Exception as e:
        print(f"Error during training: {e}")
        try:
            helpers["kill_process_group"](process)
        except Exception:
            pass

    exit_code = process.returncode if process.returncode is not None else -1

    # Find any .nam files created
    nam_files = find_nam_files(output_path)

    return exit_code, graceful_shutdown_detected, nam_files


def test_graceful_shutdown(conda_env: Optional[str] = None) -> bool:
    """
    Test that graceful shutdown generates a model file.

    :return: True if test passed (model was generated), False otherwise
    """
    print("=" * 60)
    print("Testing graceful shutdown model generation")
    print("=" * 60)

    with TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)

        # Create test data and configs
        print("\nCreating test data and configs...")
        x_path, y_path = create_test_data(tempdir)
        data_config_path, model_config_path, learning_config_path = create_configs(
            tempdir, x_path, y_path, num_epochs=100
        )

        output_path = Path(tempdir, "outputs")
        output_path.mkdir()

        print(f"Output directory: {output_path}")

        # Run training with interrupt
        print("\nStarting training (will interrupt after training starts)...")
        exit_code, graceful_detected, nam_files = run_training_with_interrupt(
            data_config_path,
            model_config_path,
            learning_config_path,
            output_path,
            interrupt_after_training_starts=True,
            interrupt_delay=2.0,  # Wait 2s after training starts
            timeout=120.0,
            conda_env=conda_env,
        )

        print("\n" + "=" * 60)
        print("Results:")
        print(f"  Exit code: {exit_code}")
        print(f"  Graceful shutdown detected: {graceful_detected}")
        print(f"  .nam files found: {len(nam_files)}")
        for f in nam_files:
            print(f"    - {f}")

        # Check output directory contents
        print("\nOutput directory contents:")
        for item in output_path.rglob("*"):
            if item.is_file():
                print(f"    {item.relative_to(output_path)}")

        print("=" * 60)

        # Test passes if we found at least one .nam file
        if nam_files:
            print("TEST PASSED: Model file was generated on graceful shutdown")
            return True
        else:
            print("TEST FAILED: No model file was generated on graceful shutdown")
            return False


class TestGracefulShutdown:
    """
    Pytest test class for graceful shutdown behavior.

    This tests that when training is interrupted with Ctrl+C (SIGINT),
    a model file is still generated from the best available checkpoint.

    Note: This test takes a few seconds to run as it needs to actually
    start training, run for a bit, and then interrupt it.
    """

    def test_graceful_shutdown_generates_model(self):
        """
        Test that graceful shutdown (Ctrl+C) generates a model file.

        This is a regression test for GitHub Issue #501:
        https://github.com/sdatkinson/neural-amp-modeler/issues/501
        """
        import pytest

        with TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)

            # Create test data and configs
            x_path, y_path = create_test_data(tempdir)
            data_config_path, model_config_path, learning_config_path = create_configs(
                tempdir, x_path, y_path, num_epochs=100
            )

            output_path = Path(tempdir, "outputs")
            output_path.mkdir()

            # Run training with interrupt
            exit_code, graceful_detected, nam_files = run_training_with_interrupt(
                data_config_path,
                model_config_path,
                learning_config_path,
                output_path,
                interrupt_after_training_starts=True,
                interrupt_delay=2.0,  # Wait 2s after training starts
                timeout=120.0,
                conda_env=None,  # Use current environment in pytest
            )

            # Assert that a .nam file was created
            assert len(nam_files) > 0, (
                f"Expected at least one .nam file to be generated on graceful shutdown, "
                f"but found {len(nam_files)}. Exit code: {exit_code}, "
                f"Graceful shutdown detected: {graceful_detected}"
            )


def main():
    """Run the graceful shutdown test."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test graceful shutdown model generation"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=None,
        help="Conda environment to use for running nam-full",
    )
    args = parser.parse_args()

    success = test_graceful_shutdown(conda_env=args.conda_env)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
