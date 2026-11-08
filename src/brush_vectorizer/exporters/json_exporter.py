"""JSON Exporter for the Stroke Vector Document (Section 2.2)."""

from __future__ import annotations

import json
from pathlib import Path
from brush_vectorizer.models import StrokeVectorDocument


def export_strokes_json(
    document: StrokeVectorDocument,
    output_path: str | Path,
    strokes_only_array: bool = False,
) -> Path:
    """Serialize StrokeVectorDocument to formatted JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if strokes_only_array:
        payload = [stroke.model_dump() for stroke in document.strokes]
    else:
        payload = document.model_dump()

    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out
