"""Meaningful CUDA contract check: precision, source binding and abstention."""
import json
from types import SimpleNamespace
import sys
import pytest
from ngsr.engines import file_sha256


def test_float_patch_preserves_bits_and_gates_abstention(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA integration check")
    from ngsr.eyes import EyeCleanPatches
    source = tmp_path / "source.exr"
    source.write_bytes(b"source identity")
    run = tmp_path / "run"
    run.mkdir()
    (run / "ngsr.json").write_text(json.dumps({"schema": "ngsr.eyes/1",
        "frames": [{"frame": 42, "index": 0, "sha256": file_sha256(source)}]}))
    patch = np.zeros((2, 2, 3), np.float32)
    patch[0, 1] = [.00001, -.00002, .00003]
    eye = SimpleNamespace(status="repaired", eye="image_right",
                          data_window=[1, 1, 3, 3], patch="patch.exr")
    abstained = SimpleNamespace(status="untouched")
    frame = SimpleNamespace(frame=0, eyes=[eye, abstained])
    class Output:
        def __init__(self, path):
            self.index = {"schema": "eyeclean.process/1"}
            self.working_color, self.size = "linear-rec709", (4, 4)
        def frames(self):
            return [frame]
        def read(self, path):
            return SimpleNamespace(channels={c: patch[..., i] for i, c in enumerate("RGB")})
    monkeypatch.setitem(sys.modules, "eyeclean_client", SimpleNamespace(RunOutput=Output))
    patches = EyeCleanPatches(run)
    original = np.full((4, 4, 3), .123456, np.float32)
    original[0, 0] = -0.
    actual, repaired, pixels = patches.apply(original, 42, source)
    keep = np.ones((4, 4), bool)
    keep[1, 2] = False
    assert np.array_equal(actual.view(np.uint32)[keep], original.view(np.uint32)[keep])
    assert np.array_equal(actual[1, 2], original[1, 2] + patch[0, 1])
    assert repaired == {"left"} and pixels == 1
    source.write_bytes(b"new key")
    with pytest.raises(ValueError, match="Stale"):
        patches.apply(original, 42, source)
