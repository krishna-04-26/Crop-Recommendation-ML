"""Set up a clone and launch the desktop app using only standard-library Python."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
ENVIRONMENT = ROOT / ".venv"
PYTHON = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(*arguments: str) -> None:
    subprocess.run([str(PYTHON), *arguments], cwd=ROOT, check=True)


def prepare() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Please install Python 3.10 or later, then open the launcher again.")
    if not PYTHON.exists():
        print("Creating the local Python environment…", flush=True)
        venv.EnvBuilder(with_pip=True).create(ENVIRONMENT)

    fingerprint = hashlib.sha256(
        (ROOT / "requirements.txt").read_bytes() + (ROOT / "pyproject.toml").read_bytes()
    ).hexdigest()
    marker = ENVIRONMENT / ".crop-setup"
    dependencies_changed = not marker.exists() or marker.read_text() != fingerprint
    if dependencies_changed:
        print("Installing dependencies. First-time setup needs internet and may take a few minutes…", flush=True)
        run("-m", "pip", "install", "-r", str(ROOT / "requirements.txt"))
        run("-m", "pip", "install", "-e", str(ROOT))
        marker.write_text(fingerprint)

    model = ROOT / "artifacts" / "decision_tree_model.joblib"
    if dependencies_changed or not model.exists():
        print("Preparing the crop model…", flush=True)
        run("-m", "crop_recommendation.train", "--data", str(ROOT / "Crop_recommendation.csv"))


def main() -> int:
    try:
        prepare()
        if "--check" in sys.argv:
            run("-c", "import PySide6.QtWidgets; import crop_recommendation; print('Ready to launch.')")
        else:
            print("Opening Crop Compass…", flush=True)
            run(str(ROOT / "desktop_app.py"))
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"\nCould not start Crop Compass: {error}", file=sys.stderr)
        print("Check that Python 3.10+ is installed and that internet is available for first-time setup.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
