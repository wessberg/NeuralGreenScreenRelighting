"""End-to-end compositor contract without a reporter fixture or bulb-tour metadata."""
from pathlib import Path
import json
import os
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_external_blender_pass_preserves_hdr_and_alpha(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA integration check")
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    cv2 = pytest.importorskip("cv2")
    shot = tmp_path / "external shot"
    for directory in ("key/fg", "key/alpha", "relight/normals", "passes"):
        (shot / directory).mkdir(parents=True)
    (shot / "shot.json").write_text(json.dumps({"source_frame_rate": "24000/1001",
        "source_resolution": [32,32], "selection": {"frame_count":1},
        "delivery":{"resolution":[32,32]}}))
    fg = np.full((32,32,3), [.2,.3,2.0], np.float32)
    alpha = np.ones((32,32), np.float32)
    alpha[:,0] = 0
    alpha[:,1] = .5
    normal = np.zeros_like(fg)
    normal[...,1] = -1
    np.savez(shot / "relight/normals/000000.npz", normal=normal)
    flags = [cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_FLOAT]
    assert cv2.imwrite(str(shot / "key/fg/000000.exr"),fg[...,::-1],flags)
    assert cv2.imwrite(str(shot / "key/alpha/000000.exr"),alpha,flags)
    irradiance = np.full((32,32,4),.6,np.float32)
    irradiance[...,3] = 1
    assert cv2.imwrite(str(shot / "passes/000000.exr"),irradiance,flags)
    subprocess.run([sys.executable,"-B",str(ROOT/"scripts/composite_ratio.py"),
        "--shot",str(shot),"--render-dir",str(shot/"passes"),"--name","result",
        "--specular","0","--linear-output"],check=True,cwd=ROOT)
    result = cv2.imread(str(shot / "composite/result/000000.exr"),-1)
    assert result.shape == (32,32,4)
    np.testing.assert_allclose(result[...,3],alpha,rtol=0,atol=1e-7)
    np.testing.assert_allclose(result[8,8,:3],fg[8,8,::-1],rtol=0,atol=2e-5)
    assert result[8,8,0] > 1.9
    assert np.count_nonzero(result[:,0,:3]) == 0
    manifest = json.loads((shot / "composite/result/composite_manifest.json").read_text())
    assert manifest["render_manifest"] is None
    assert manifest["linear_color"] == "linear-rec709"
