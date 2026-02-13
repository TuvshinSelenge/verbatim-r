import json
from pathlib import Path
from typing import Any, List


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text())


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_report_lines(lines: List[str], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines))
