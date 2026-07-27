"""Modal runner for the PyTorch time-series pipeline.

Examples:

    source /home/aria/sharif/DS/DS_HW4/ml-env/bin/activate
    modal run modal_app.py --smoke-only
    modal run modal_app.py --full-only --epochs 20 --baseline seasonal_mad

The local CSV files are included in the image, copied to the persistent
``rexi`` volume, and all run outputs are written below ``/mnt/rexi/runs``.
"""

from __future__ import annotations

import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import modal


APP_NAME = "thanos-timeseries-pytorch"
VOLUME_NAME = "rexi"
VOLUME_MOUNT = "/mnt/rexi"
DATA_MOUNT = f"{VOLUME_MOUNT}/data"
RUNS_MOUNT = f"{VOLUME_MOUNT}/runs"
IMAGE_ROOT = "/opt/rexi"

# This image already contains CUDA-enabled PyTorch. L40S is selected on the
# functions below, so no CUDA toolkit or GPU wheel is installed at runtime.
PYTORCH_IMAGE = "pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime"

image = (
    modal.Image.from_registry(PYTORCH_IMAGE)
    .entrypoint([])
    .uv_pip_install("pandas==3.0.3", "matplotlib==3.10.3")
    .add_local_dir("src", remote_path=f"{IMAGE_ROOT}/src")
    .add_local_file("final.csv", remote_path=f"{IMAGE_ROOT}/final.csv")
    .add_local_file(
        "metadata-hashed.csv",
        remote_path=f"{IMAGE_ROOT}/metadata-hashed.csv",
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _run_directory(mode: str, seed: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(RUNS_MOUNT) / f"{timestamp}_{mode}_seed{seed}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _cuda_info() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Modal L40S function.")
    probe = torch.ones((8, 8), device="cuda")
    torch.cuda.synchronize()
    return {
        "cuda_available": True,
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "probe_sum": float((probe @ probe).sum().item()),
    }


def _ensure_data(refresh: bool = False) -> dict[str, str]:
    import shutil

    volume_data = Path(DATA_MOUNT)
    volume_data.mkdir(parents=True, exist_ok=True)
    sources = {
        "final.csv": Path(IMAGE_ROOT) / "final.csv",
        "metadata-hashed.csv": Path(IMAGE_ROOT) / "metadata-hashed.csv",
    }
    for name, source in sources.items():
        destination = volume_data / name
        if refresh or not destination.exists():
            shutil.copy2(source, destination)
    volume.commit()
    return {name: str(volume_data / name) for name in sources}


def _run_command(
    command: list[str],
    run_dir: Path,
    manifest: dict[str, Any],
    log_name: str = "run.log",
) -> dict[str, Any]:
    import os
    import subprocess
    import sys
    import traceback

    log_path = run_dir / log_name
    manifest_path = run_dir / "manifest.json"
    manifest["command"] = command
    manifest["command_shell"] = shlex.join(command)
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    volume.commit()

    environment = os.environ.copy()
    environment["PYTHONPATH"] = IMAGE_ROOT
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            process = subprocess.Popen(
                command,
                cwd=IMAGE_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
            return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        manifest.update({"status": "ok", "return_code": return_code})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        volume.commit()
        return {"status": "ok", "run_dir": str(run_dir), "manifest": manifest}
    except Exception as error:
        manifest.update(
            {
                "status": "error",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        volume.commit()
        raise


def _experiment(
    mode: str,
    seed: int,
    baseline: str,
    architecture: str,
    epochs: int,
    learning_rate: float,
    use_metadata: bool,
    refresh_data: bool,
) -> dict[str, Any]:
    import sys

    if learning_rate <= 0:
        learning_rate = 3e-3 if architecture == "panel_factor" else 3e-4
    cuda = _cuda_info()
    data = _ensure_data(refresh=refresh_data)
    run_dir = _run_directory(mode, seed)
    manifest: dict[str, Any] = {
        "mode": mode,
        "seed": seed,
        "baseline": baseline,
        "architecture": architecture,
        "use_metadata": use_metadata,
        "cuda": cuda,
        "data": data,
        "run_dir": str(run_dir),
    }

    # Both smoke and full runs produce the full-week baseline anomaly report.
    baseline_command = [
        sys.executable,
        "-u",
        "-m",
        "src.run_baselines",
        "--csv",
        data["final.csv"],
        "--method",
        "both",
        "--output",
        str(run_dir / "baselines.pt"),
        "--report-dir",
        str(run_dir / "baseline_report"),
    ]
    _run_command(baseline_command, run_dir, manifest, log_name="baselines.log")

    common_command = [
        "--csv",
        data["final.csv"],
        "--baseline",
        baseline,
        "--epochs",
        str(1 if mode == "smoke" else epochs),
        "--learning-rate",
        str(learning_rate),
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--output-dir",
        str(run_dir / "training"),
    ]
    if architecture == "panel_factor":
        command = [
            sys.executable,
            "-u",
            "-m",
            "src.train_panel",
            *common_command,
            "--rank",
            "8" if mode == "smoke" else "32",
            "--time-batch-size",
            "64",
            "--inference-steps",
            "5" if mode == "smoke" else "60",
            "--inference-learning-rate",
            "0.05",
            "--collective-weight",
            "0.5",
        ]
    else:
        command = [
            sys.executable,
            "-u",
            "-m",
            "src.train",
            *common_command,
            "--architecture",
            architecture,
            "--num-workers",
            "0",
            "--full-score",
        ]
        if mode == "smoke":
            command += [
                "--hidden-size",
                "32",
                "--window",
                "32",
                "--batch-size",
                "32",
                "--samples-per-epoch",
                "256",
                "--evaluation-samples",
                "256",
                "--patience",
                "1",
                "--score-max-time-points",
                "24",
                "--score-max-series",
                "128",
                "--score-batch-size",
                "128",
            ]
        elif architecture == "context_tcn":
            command += [
                "--hidden-size",
                "96",
                "--tcn-blocks",
                "4",
                "--window",
                "96",
                "--batch-size",
                "384",
                "--samples-per-epoch",
                "200000",
                "--evaluation-samples",
                "200000",
                "--dropout",
                "0.1",
                "--weight-decay",
                "0.0005",
                "--patience",
                "8",
                "--score-batch-size",
                "1024",
            ]
    if use_metadata:
        command += ["--metadata", data["metadata-hashed.csv"]]

    result = _run_command(command, run_dir, manifest, log_name="training.log")
    metrics_path = run_dir / "training" / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Expected training metrics were not written: {metrics_path}")
    result["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return result


@app.function(
    gpu="L40S",
    cpu=0.125,
    memory=258,
    timeout=10 * 60,
)
def check_environment() -> dict[str, Any]:
    """Smoke-check the L40S and the CUDA-enabled PyTorch image."""
    return _cuda_info()


@app.function(
    gpu="L40S",
    cpu=0.125,
    memory=258,
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: volume},
)
def smoke(
    seed: int = 17,
    baseline: str = "seasonal_mad",
    architecture: str = "panel_factor",
    use_metadata: bool = True,
    refresh_data: bool = False,
) -> dict[str, Any]:
    return _experiment(
        "smoke",
        seed,
        baseline,
        architecture,
        epochs=1,
        learning_rate=3e-3 if architecture == "panel_factor" else 1e-3,
        use_metadata=use_metadata,
        refresh_data=refresh_data,
    )


@app.function(
    gpu="L40S",
    cpu=0.125,
    memory=258,
    timeout=24 * 60 * 60,
    volumes={VOLUME_MOUNT: volume},
)
def full(
    seed: int = 17,
    baseline: str = "seasonal_mad",
    architecture: str = "panel_factor",
    epochs: int = 20,
    learning_rate: float = 0.0,
    use_metadata: bool = True,
    refresh_data: bool = False,
) -> dict[str, Any]:
    return _experiment(
        "full",
        seed,
        baseline,
        architecture,
        epochs,
        learning_rate,
        use_metadata,
        refresh_data,
    )


@app.local_entrypoint()
def main(
    smoke_only: bool = False,
    full_only: bool = False,
    seed: int = 17,
    baseline: str = "seasonal_mad",
    architecture: str = "panel_factor",
    epochs: int = 20,
    learning_rate: float = 0.0,
    use_metadata: bool = True,
    refresh_data: bool = False,
) -> None:
    if smoke_only and full_only:
        raise ValueError("Choose only one of --smoke-only or --full-only.")
    if baseline not in {"seasonal_mad", "holt_winters", "none"}:
        raise ValueError("--baseline must be seasonal_mad, holt_winters, or none.")
    if architecture not in {"gru", "context_tcn", "panel_factor"}:
        raise ValueError(
            "--architecture must be gru, context_tcn, or panel_factor."
        )
    if epochs < 1:
        raise ValueError("--epochs must be positive.")
    if learning_rate < 0:
        raise ValueError("--learning-rate cannot be negative.")

    # Default invocation is deliberately smoke-only; full training is opt-in.
    if not full_only:
        result = smoke.remote(
            seed=seed,
            baseline=baseline,
            architecture=architecture,
            use_metadata=use_metadata,
            refresh_data=refresh_data,
        )
        print("Smoke result:", json.dumps(result, indent=2, default=str))
        if smoke_only:
            return
        return

    result = full.remote(
        seed=seed,
        baseline=baseline,
        architecture=architecture,
        epochs=epochs,
        learning_rate=learning_rate,
        use_metadata=use_metadata,
        refresh_data=refresh_data,
    )
    print("Full result:", json.dumps(result, indent=2, default=str))
    print(f"Persistent results: modal volume ls {VOLUME_NAME} runs")


if __name__ == "__main__":
    raise SystemExit("Use: modal run modal_app.py --smoke-only")
