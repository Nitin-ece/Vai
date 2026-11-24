"""Tests for the FastAPI endpoints."""
from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from mangavrse.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_list_stages(client: TestClient):
    resp = client.get("/api/v1/stages")
    assert resp.status_code == 200
    stages = resp.json()
    assert "noop" in stages
    assert "ingest" in stages
    assert "enhance" in stages
    assert "vectorize" in stages
    assert "segment" in stages
    assert "layers" in stages
    assert "colorize" in stages
    assert "depth" in stages
    assert "assembly" in stages


def test_run_colorize_stage(client: TestClient, tmp_path: Path):
    inp = tmp_path / "col_in"
    out = tmp_path / "col_out"
    inp.mkdir()

    from tests.test_colorize import create_synthetic_panel_for_colorize
    create_synthetic_panel_for_colorize(inp, 150, 150)

    resp = client.post("/api/v1/stages/colorize/run", json={
        "input_dir": str(inp),
        "output_dir": str(out),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["cached"] is False
    assert (out / "colored_panel.svg").exists()
    assert (out / "colored_panel.png").exists()
    assert (out / "comparison.png").exists()
    assert (out / "palette.json").exists()


def test_colorize_override_endpoint(client: TestClient, tmp_path: Path):
    from tests.test_colorize import create_synthetic_panel_for_colorize
    create_synthetic_panel_for_colorize(tmp_path, 150, 150)

    resp = client.post("/api/v1/stages/colorize/override", json={
        "panel_dir": str(tmp_path),
        "layer_id": "seg_hair",
        "color": "#00FF00",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["color"] == "#00FF00"
    assert Path(data["layer_svg_path"]).exists()


def test_run_layers_stage(client: TestClient, tmp_path: Path):
    inp = tmp_path / "lyr_in"
    out = tmp_path / "lyr_out"
    inp.mkdir()

    from tests.test_layers import create_synthetic_panel_and_segments
    create_synthetic_panel_and_segments(inp, 150, 150)

    resp = client.post("/api/v1/stages/layers/run", json={
        "input_dir": str(inp),
        "output_dir": str(out),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["cached"] is False
    assert (out / "background_filled.png").exists()
    assert (out / "parallax_test.png").exists()
    assert (out / "layers_manifest.json").exists()


def test_inpaint_region_endpoint(client: TestClient, tmp_path: Path):
    img_p = tmp_path / "base.png"
    import numpy as np
    import cv2
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (0, 0, 0), -1)
    cv2.imwrite(str(img_p), img)

    out_p = tmp_path / "filled.png"
    resp = client.post("/api/v1/stages/layers/inpaint-region", json={
        "image_path": str(img_p),
        "box": [30, 30, 70, 70],
        "output_path": str(out_p),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert out_p.exists()
    assert "metrics" in data


def test_run_segment_stage(client: TestClient, tmp_path: Path):
    inp = tmp_path / "seg_in"
    out = tmp_path / "seg_out"
    inp.mkdir()

    import numpy as np
    import cv2
    img = np.full((120, 120, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (110, 110), (0, 0, 0), 2)
    cv2.circle(img, (60, 60), 25, (50, 50, 50), -1)
    cv2.imwrite(str(inp / "enhanced.png"), img)

    resp = client.post("/api/v1/stages/segment/run", json={
        "input_dir": str(inp),
        "output_dir": str(out),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["cached"] is False
    assert (out / "segments.json").exists()
    assert (out / "overlay.png").exists()


def test_refine_segment_endpoint(client: TestClient, tmp_path: Path):
    p_img = tmp_path / "refine_panel.png"
    import numpy as np
    import cv2
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (0, 0, 0), -1)
    cv2.imwrite(str(p_img), img)

    out_m = tmp_path / "out_mask.png"
    resp = client.post("/api/v1/stages/segment/refine", json={
        "image_path": str(p_img),
        "points": [{"x": 50, "y": 50, "label": 1}],
        "output_mask_path": str(out_m),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert out_m.exists()
    assert data["area"] > 0
    assert data["confidence"] > 0.5


def test_run_vectorize_stage(client: TestClient, tmp_path: Path):
    inp = tmp_path / "vec_in"
    out = tmp_path / "vec_out"
    inp.mkdir()

    import numpy as np
    import cv2
    img = np.full((120, 120, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (110, 110), (0, 0, 0), 3)
    cv2.imwrite(str(inp / "enhanced.png"), img)

    resp = client.post("/api/v1/stages/vectorize/run", json={
        "input_dir": str(inp),
        "output_dir": str(out),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["cached"] is False
    assert "panel/lines_svg" in data["artifacts"]
    assert "panel/tones_svg" in data["artifacts"]


def test_run_enhance_stage(client: TestClient, tmp_path: Path):
    inp = tmp_path / "panel_in"
    out = tmp_path / "panel_out"
    inp.mkdir()

    import numpy as np
    import cv2
    img = np.full((100, 100, 3), 250, dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (0, 0, 0), -1)
    cv2.imwrite(str(inp / "original.png"), img)

    resp = client.post("/api/v1/stages/enhance/run", json={
        "input_dir": str(inp),
        "output_dir": str(out),
        "config_overrides": {
            "enhance": {
                "upscale": {"scale": 2, "tile_size": 256, "tile_pad": 16}
            }
        }
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["cached"] is False
    assert "panel/enhanced" in data["artifacts"]
    assert "panel/comparison" in data["artifacts"]


def test_run_ingest_stage(client: TestClient, tmp_path: Path):
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()

    # Create synthetic page
    import numpy as np
    import cv2
    img = np.full((1000, 800, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (750, 450), (0, 0, 0), 4)
    cv2.rectangle(img, (50, 500), (750, 950), (0, 0, 0), 4)
    cv2.imwrite(str(inp / "page_001.png"), img)

    resp = client.post("/api/v1/stages/ingest/run", json={
        "input_dir": str(inp),
        "output_dir": str(out),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["cached"] is False
    assert "page_001/panels_json" in data["artifacts"]


def test_run_noop_stage(client: TestClient, tmp_path: Path):
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    (inp / "test.txt").write_text("hello", encoding="utf-8")

    resp = client.post("/api/v1/stages/noop/run", json={
        "input_dir": str(inp),
        "output_dir": str(out),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["cached"] is False
    assert "test.txt" in data["artifacts"]


def test_run_noop_cached_on_second_call(client: TestClient, tmp_path: Path):
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    (inp / "test.txt").write_text("cache me", encoding="utf-8")

    resp1 = client.post("/api/v1/stages/noop/run", json={
        "input_dir": str(inp),
        "output_dir": str(out),
    })
    assert resp1.status_code == 200
    assert resp1.json()["cached"] is False

    resp2 = client.post("/api/v1/stages/noop/run", json={
        "input_dir": str(inp),
        "output_dir": str(out),
    })
    assert resp2.status_code == 200
    assert resp2.json()["cached"] is True


def test_run_unknown_stage(client: TestClient, tmp_path: Path):
    resp = client.post("/api/v1/stages/nonexistent/run", json={
        "input_dir": str(tmp_path),
        "output_dir": str(tmp_path / "out"),
    })
    assert resp.status_code == 404


def test_run_depth_stage(client: TestClient, tmp_path: Path):
    from tests.test_depth import create_synthetic_panel_for_depth
    inp = tmp_path / "depth_in"
    out = tmp_path / "depth_api_out"
    inp.mkdir()
    p_in = inp / "panel_001"
    create_synthetic_panel_for_depth(p_in)

    resp = client.post("/api/v1/stages/depth/run", json={
        "input_dir": str(inp),
        "output_dir": str(out),
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert (out / "panel_001" / "depth_map.png").exists()
    assert (out / "panel_001" / "depth.json").exists()
    assert (out / "panel_001" / "depth_visualization.png").exists()


def test_depth_override_endpoint(client: TestClient, tmp_path: Path):
    from tests.test_depth import create_synthetic_panel_for_depth
    inp = tmp_path / "depth_override_in"
    out = tmp_path / "depth_api_override_out"
    inp.mkdir()
    p_in = inp / "panel_001"
    create_synthetic_panel_for_depth(p_in)

    # Run first
    client.post("/api/v1/stages/depth/run", json={
        "input_dir": str(inp),
        "output_dir": str(out),
    })

    p_dir = out / "panel_001"
    resp = client.post("/api/v1/stages/depth/override", json={
        "panel_dir": str(p_dir),
        "layer_id": "lyr_bg",
        "depth": 0.88,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["layer_id"] == "lyr_bg"
    assert data["depth"] == 0.88


def test_scenes_endpoints(client: TestClient, tmp_path: Path):
    # Test listing scenes (initially empty or populated)
    resp = client.get("/api/v1/scenes")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # Test 404 for unknown scene
    resp_404 = client.get("/api/v1/scenes/nonexistent_panel_xyz")
    assert resp_404.status_code == 404


def test_system_and_project_endpoints(client: TestClient):
    # GPU status
    resp_gpu = client.get("/api/v1/system/gpu")
    assert resp_gpu.status_code == 200
    assert "platform" in resp_gpu.json()

    # Projects list
    resp_proj = client.get("/api/v1/projects")
    assert resp_proj.status_code == 200
    assert isinstance(resp_proj.json(), list)

    # Demo project
    resp_demo = client.post("/api/v1/projects/demo")
    assert resp_demo.status_code == 200
    assert resp_demo.json()["success"] is True

    # Pages for benchmark
    resp_pages = client.get("/api/v1/projects/benchmark/pages")
    assert resp_pages.status_code == 200
    assert len(resp_pages.json()) >= 1

    # Stages for benchmark panel
    resp_stages = client.get("/api/v1/projects/benchmark/panels/clean_lineart/stages")
    assert resp_stages.status_code == 200
    data = resp_stages.json()
    assert "stages" in data
    assert "enhance" in data["stages"]



