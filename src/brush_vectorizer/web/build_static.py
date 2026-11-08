"""Static site builder packaging Brush Vectorizer Studio for Pure Client-Side WebAssembly (Pyodide)."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import zipfile


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


def build_python_bundle_zip(zip_path: Path) -> Path:
    """Package the brush_vectorizer Python modules needed by Pyodide into a compact .zip archive."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_files = {"build_static.py"}

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for py_file in sorted(PACKAGE_ROOT.rglob("*.py")):
            if "__pycache__" in py_file.parts or "static" in py_file.parts:
                continue
            if py_file.name in excluded_files:
                continue
            arcname = Path("brush_vectorizer") / py_file.relative_to(PACKAGE_ROOT)
            zf.write(py_file, arcname=str(arcname))

    return zip_path


def build_static_site(
    output_dir: str | Path = "dist",
    workspace_dir: str | Path | None = None,
) -> Path:
    """Build a self-contained static website directory ready for any static host (GitHub Pages, Netlify, etc.)."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ws = Path(workspace_dir).resolve() if workspace_dir else Path.cwd()

    # 1. Build brush_vectorizer.zip directly into output_dir
    build_python_bundle_zip(out / "brush_vectorizer.zip")

    # 2. Copy index.html and pyodide_worker.js
    for asset_name in ("index.html", "pyodide_worker.js"):
        src_asset = STATIC_DIR / asset_name
        dst_asset = out / asset_name
        if src_asset != dst_asset:
            shutil.copy2(src_asset, dst_asset)

    # 3. Populate static sample images & manifest.json from STATIC_DIR/samples and workspace_dir
    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_files: list[str] = []

    source_dirs = [STATIC_DIR / "samples", ws]
    for folder in source_dirs:
        if not folder.is_dir():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img_path in sorted(folder.glob(ext)):
                if img_path.name.endswith(".painterly.png") or "debug" in img_path.name:
                    continue
                if img_path.name in sample_files:
                    continue
                dst_img = samples_dir / img_path.name
                if img_path.resolve() != dst_img.resolve():
                    shutil.copy2(img_path, dst_img)
                sample_files.append(img_path.name)

    manifest_path = samples_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"files": sample_files}, indent=2),
        encoding="utf-8",
    )

    return out


def serve_static_site(directory: Path, host: str = "127.0.0.1", port: int = 8080) -> None:
    """Serve the built static site directory using Python's standard-library HTTP server."""
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    with ThreadingHTTPServer((host, port), handler) as httpd:
        print(f"\nBrush Vectorizer Studio (WASM) running at: http://{host}:{port}\nPress Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build (and optionally serve) the Pure Client-Side WebAssembly static site."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="dist",
        help="Target directory for the static site bundle (default: dist).",
    )
    parser.add_argument(
        "-s",
        "--serve",
        action="store_true",
        help="Start a local static HTTP server after building.",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8080,
        help="Port for --serve (default: 8080).",
    )
    args = parser.parse_args()
    built = build_static_site(output_dir=args.output_dir)
    print(f"Built static Pyodide site in: {built}")
    if args.serve:
        serve_static_site(built, port=args.port)


if __name__ == "__main__":
    main()
