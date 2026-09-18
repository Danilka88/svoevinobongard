# START_MODULE_CONTRACT
# PURPOSE: Верификация эвалюатора — реальный multipart-POST, парсинг slug и top5, latency, метрики
# SCOPE: test_module_is_stdlib_only, test_slug_parsing_both_formats, test_top5_parsing, test_metrics_perfect_and_partial, test_percentile_monotonic, test_real_multipart_post_and_report
# DEPENDS: M-EVAL (scripts/eval/evaluate.py)
# LINKS: M-EVAL
# ROLE: TEST
# MAP_MODE: LOCALS
# END_MODULE_CONTRACT
# START_MODULE_MAP
# ROOT - корень проекта
# EVAL_PY - путь к scripts/eval/evaluate.py
# ev - загруженный модуль эвалюатора
# stub_endpoint - фикстура локального stub-сервера
# test_module_is_stdlib_only - отсутствие внешних зависимостей
# test_slug_parsing_both_formats - парсинг объекта и массива
# test_top5_parsing - парсинг dict- и str-элементов
# test_metrics_perfect_and_partial - accuracy@1/@5 и macro-F1
# test_percentile_monotonic - монотонность p95 и p50
# test_real_multipart_post_and_report - прогон против stub-сервера
# END_MODULE_MAP
"""V-M-002: офлайн-эвалюатор.

Ключевой сценарий — реальный HTTP multipart-POST против локального stub-сервера
(без внешней сети), как это делает participant_test.sh организатора.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_PY = ROOT / "scripts" / "eval" / "evaluate.py"

# 1x1 PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("svoe_evaluate", EVAL_PY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ev = _load_eval_module()


class _StubHandler(BaseHTTPRequestHandler):
    last_body: bytes = b""
    last_content_type: str = ""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        _StubHandler.last_body = self.rfile.read(length)
        _StubHandler.last_content_type = self.headers.get("Content-Type", "")
        payload = json.dumps(
            {"slug": "stub-slug", "top5": [{"slug": "stub-slug"}, {"slug": "other-slug"}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # silence
        return


@pytest.fixture(scope="module")
def stub_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1/scan"
    finally:
        server.shutdown()
        server.server_close()


def test_module_is_stdlib_only() -> None:
    import ast

    src = EVAL_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    stdlib = {"argparse", "json", "mimetypes", "os", "sys", "time", "urllib",
              "uuid", "collections", "pathlib", "__future__"}
    assert imported <= stdlib, f"внешние зависимости: {sorted(imported - stdlib)}"


def test_slug_parsing_both_formats() -> None:
    assert ev.extract_slug({"slug": "a"}) == "a"
    assert ev.extract_slug([{"slug": "b"}]) == "b"
    assert ev.extract_slug({"slug": ""}) is None
    assert ev.extract_slug({}) is None
    assert ev.extract_slug(None) is None


def test_top5_parsing() -> None:
    data = {"top5": [{"slug": "a", "score": 0.9}, {"slug": "b", "score": 0.5}, "c"]}
    assert ev.extract_top5(data) == ["a", "b", "c"]
    assert ev.extract_top5({}) == []
    assert ev.extract_top5(None) == []


def test_metrics_perfect_and_partial() -> None:
    perfect = [{"query_id": "q1", "predicted_slug": "a", "top5": ["a", "b"]},
               {"query_id": "q2", "predicted_slug": "b", "top5": ["b", "c"]}]
    gt = {"q1": "a", "q2": "b"}
    m = ev.compute_metrics(perfect, gt)
    assert m["accuracy@1"] == 1.0
    assert m["accuracy@5"] == 1.0
    assert m["f1_macro@1"] == 1.0

    partial = [{"query_id": "q1", "predicted_slug": "a", "top5": ["a", "b"]},
               {"query_id": "q2", "predicted_slug": "z", "top5": ["b", "c"]}]
    m2 = ev.compute_metrics(partial, gt)
    assert m2["accuracy@1"] == 0.5
    # q2: gold 'b' попал в top5 -> засчитан
    assert m2["accuracy@5"] == 1.0


def test_percentile_monotonic() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert ev.percentile(values, 0.5) == 30.0
    assert ev.percentile(values, 0.95) >= ev.percentile(values, 0.5)
    assert ev.percentile([], 0.5) == 0.0


def test_real_multipart_post_and_report(tmp_path: Path, stub_endpoint: str) -> None:
    """Реальный subprocess-прогон эвалюатора против stub-сервера."""
    (tmp_path / "q1.png").write_bytes(_PNG)
    out = tmp_path / "predictions.jsonl"

    proc = subprocess.run(
        [sys.executable, str(EVAL_PY), "--images", str(tmp_path),
         "--endpoint", stub_endpoint, "--output", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["predicted_slug"] == "stub-slug"
    assert rows[0]["top5"] == ["stub-slug", "other-slug"]
    assert rows[0]["latency_ms"] >= 0

    # multipart-контракт организатора: поле image + multipart/form-data
    assert b'name="image"' in _StubHandler.last_body
    assert _StubHandler.last_content_type.startswith("multipart/form-data; boundary=")

    report = json.loads((tmp_path / "predictions.report.json").read_text(encoding="utf-8"))
    assert report["processed"] == 1
    assert report["null_slug"] == 0
    assert report["latency_ms"]["p95"] >= report["latency_ms"]["p50"]
