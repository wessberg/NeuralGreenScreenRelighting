import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from ngsr import run_eyeclean, run_keyer, run_stage
from ngsr.engines import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def test_keyer_argv_and_linear_contract(tmp_path, monkeypatch):
    input_dir = tmp_path / "frames with spaces"
    input_dir.mkdir()
    output = tmp_path / "result"
    calls = []
    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        (output / "frames_alpha_exr").mkdir(parents=True)
        (output / "frames_fg_exr").mkdir()
    monkeypatch.setattr("ngsr.engines.subprocess.run", execute)
    a, fg = run_keyer(input_dir, output, repository=tmp_path, python=sys.executable,
                       fps="24000/1001", max_frames=8, ffmpeg="ffmpeg executable")
    command, options = calls[0]
    assert command[command.index("--input")+1] == str(input_dir)
    assert command[command.index("--output_format")+1] == "exr"
    assert command[command.index("--exr_dtype")+1] == "float"
    assert command[command.index("--despill_strength")+1] == "0"
    assert options["check"] is True and "shell" not in options
    assert json.loads((output / "ngsr.json").read_text())["fps"] == "24000/1001"
    assert a.name == "frames_alpha_exr" and fg.name == "frames_fg_exr"


def test_stage_failure_is_not_swallowed(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/neural_normals.py").touch()
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(3, args[0])
    monkeypatch.setattr("ngsr.stages.subprocess.run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_stage("normals", tmp_path, repository=tmp_path)


def test_external_shot_and_exact_rate(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("shot_paths", ROOT / "scripts/shot_paths.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path / "independent-checkout")
    (tmp_path / "shot.json").write_text(json.dumps({"source_frame_rate": "60000/1001",
        "selection": {"frame_count": 3}, "environment": {"dir": "environment"}}))
    shot = module.Shot(tmp_path)
    assert str(shot.frame_rate) == "60000/1001"
    assert shot.environment == tmp_path / "environment"
    assert shot.frame(2) == tmp_path / "source/frames/000002.png"


def test_eyeclean_binding_and_colour(tmp_path, monkeypatch):
    fg = tmp_path / "fg"
    fg.mkdir()
    (fg / "000000.exr").write_bytes(b"mock EXR")
    output = tmp_path / "eyes"
    def run(job, **kwargs):
        assert job["working_color"] == job["input_color"] == "linear-rec709"
        assert job["fps"] == "60000/1001"
        output.mkdir()
        return SimpleNamespace(report={"frames": 1, "model": {"sha256": "model"},
                                       "eyeclean_version": "0.4.0"})
    monkeypatch.setitem(sys.modules, "eyeclean_client", SimpleNamespace(run=run))
    run_eyeclean(fg, output, fps="60000/1001")
    binding = json.loads((output / "ngsr.json").read_text())
    assert binding["frames"][0]["sha256"] == file_sha256(fg / "000000.exr")
    assert binding["schema"] == "ngsr.eyes/1"
    with pytest.raises(FileExistsError):
        run_eyeclean(fg, output, fps=24)


def test_eyeclean_source_mutation_rejects_completion(tmp_path, monkeypatch):
    fg = tmp_path / "fg"
    fg.mkdir()
    source = fg / "000000.exr"
    source.write_bytes(b"original")
    output = tmp_path / "eyes"
    def run(*args, **kwargs):
        source.write_bytes(b"modified")
        output.mkdir()
        return SimpleNamespace(report={"frames": 1})
    monkeypatch.setitem(sys.modules, "eyeclean_client", SimpleNamespace(run=run))
    with pytest.raises(ValueError, match="Source changed"):
        run_eyeclean(fg, output, fps=24)
    assert not (output / "ngsr.json").exists()


@pytest.mark.parametrize("fps", [0, -24])
def test_invalid_timing(tmp_path, fps):
    with pytest.raises(ValueError):
        run_keyer(tmp_path, tmp_path / "out", repository=tmp_path,
                   python=sys.executable, fps=fps)


def test_sequence_gaps_fail_before_engine(tmp_path, monkeypatch):
    fg = tmp_path / "fg"
    fg.mkdir()
    (fg / "000000.exr").write_bytes(b"one")
    (fg / "000002.exr").write_bytes(b"three")
    def forbidden(*args, **kwargs):
        pytest.fail("Engine must not run on a discontinuous sequence")
    monkeypatch.setitem(sys.modules, "eyeclean_client", SimpleNamespace(run=forbidden))
    with pytest.raises(ValueError, match="contiguous"):
        run_eyeclean(fg, tmp_path / "out", fps=24)
