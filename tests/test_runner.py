"""Tests for the Docker runner (argv construction + subprocess invocation)."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pytest

from plms.exceptions import ImageNotFoundError, RunnerError
from plms.runner import RunSpec, SubprocessDockerRunner, build_argv


def _spec(tmp_path: Path, **overrides) -> RunSpec:
    defaults = dict(
        image="plms-esm2:t6_8M",
        command=["embed", "--input", "/in/seqs.fasta", "--output", "/out"],
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
    )
    defaults.update(overrides)
    return RunSpec(**defaults)


def test_build_argv_cpu(tmp_path: Path) -> None:
    argv = build_argv(_spec(tmp_path))
    assert "--gpus" not in argv
    assert argv[:3] == ["docker", "run", "--rm"]
    assert f"{tmp_path / 'in'}:/in:ro" in argv
    assert f"{tmp_path / 'out'}:/out:rw" in argv
    # command comes after the image
    assert argv[argv.index("plms-esm2:t6_8M") + 1 :] == [
        "embed",
        "--input",
        "/in/seqs.fasta",
        "--output",
        "/out",
    ]


def test_build_argv_gpu_inserts_gpus_all(tmp_path: Path) -> None:
    argv = build_argv(_spec(tmp_path, use_gpu=True))
    i = argv.index("--gpus")
    assert argv[i + 1] == "all"


def test_build_argv_makes_relative_mounts_absolute(tmp_path: Path, monkeypatch) -> None:
    # Relative paths in `docker run -v` are interpreted as named volumes, not
    # host bind-mounts, so mount sources must be absolute.
    monkeypatch.chdir(tmp_path)
    spec = RunSpec(image="img", command=["manifest"], input_dir=Path("in"), output_dir=Path("out"))
    argv = build_argv(spec)
    assert f"{tmp_path / 'in'}:/in:ro" in argv
    assert f"{tmp_path / 'out'}:/out:rw" in argv


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX-only user mapping")
def test_build_argv_runs_as_current_user(tmp_path: Path) -> None:
    # Map the container to the host user so outputs are not owned by root.
    argv = build_argv(_spec(tmp_path))
    assert "--user" in argv
    assert f"{os.getuid()}:{os.getgid()}" in argv


def test_build_argv_includes_sorted_env(tmp_path: Path) -> None:
    argv = build_argv(_spec(tmp_path, extra_env={"B": "2", "A": "1"}))
    # env flags appear in sorted key order
    assert "-e" in argv
    a_idx = argv.index("A=1")
    b_idx = argv.index("B=2")
    assert a_idx < b_idx


def test_build_argv_respects_custom_executable(tmp_path: Path) -> None:
    argv = build_argv(_spec(tmp_path), docker_executable="podman")
    assert argv[0] == "podman"


def test_run_returns_result_and_records_argv(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_run(argv, capture_output, text, check):  # noqa: ANN001
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = SubprocessDockerRunner().run(_spec(tmp_path))
    assert result.exit_code == 0
    assert result.stdout == "ok"
    assert result.argv == captured["argv"]


def test_run_does_not_raise_on_nonzero_exit(tmp_path: Path, monkeypatch) -> None:
    def fake_run(argv, capture_output, text, check):  # noqa: ANN001
        return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = SubprocessDockerRunner().run(_spec(tmp_path))
    assert result.exit_code == 1
    assert result.stderr == "boom"


def test_run_raises_runner_error_when_docker_missing(tmp_path: Path, monkeypatch) -> None:
    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RunnerError):
        SubprocessDockerRunner().run(_spec(tmp_path))


def test_run_logs_argv(tmp_path: Path, monkeypatch, caplog) -> None:
    def fake_run(argv, capture_output, text, check):  # noqa: ANN001
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level(logging.INFO, logger="plms.runner"):
        SubprocessDockerRunner().run(_spec(tmp_path))
    assert any("docker" in rec.getMessage() for rec in caplog.records)


def test_manifest_runs_manifest_subcommand(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_run(argv, capture_output, text, check):  # noqa: ANN001
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, returncode=0, stdout='{"x": 1}', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = SubprocessDockerRunner().manifest("plms-esm2:t6_8M")
    assert out == '{"x": 1}'
    assert captured["argv"] == ["docker", "run", "--rm", "plms-esm2:t6_8M", "manifest"]


def test_manifest_nonzero_exit_raises_image_not_found(tmp_path: Path, monkeypatch) -> None:
    def fake_run(argv, capture_output, text, check):  # noqa: ANN001
        return subprocess.CompletedProcess(
            argv, returncode=125, stdout="", stderr="Unable to find image locally"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ImageNotFoundError):
        SubprocessDockerRunner().manifest("plms-esm2:missing")
