import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_performance_page_matches_recorded_results():
    spec = importlib.util.spec_from_file_location(
        "render_performance_tables", ROOT / "tools/render_performance_tables.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    page = (ROOT / "docs/performance.md").read_text(encoding="utf-8")
    assert page == module.render(), "Run python tools/render_performance_tables.py"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme == module.readme_text(), "Run python tools/render_performance_tables.py"
