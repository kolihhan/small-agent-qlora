from __future__ import annotations

from pathlib import Path
import zipfile


def detect_content_type(path: str | Path) -> str:
    """Detect only the file types the local tools explicitly support.

    GAIA attachments can be materialized without their original suffix, so the
    reader cannot rely on ``Path.suffix`` alone. Unknown ZIP containers remain
    ``zip`` and are not treated as readable text.
    """
    path = Path(path)
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".xlsx", ".xlsm"}:
        return "xlsx"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"

    try:
        with path.open("rb") as handle:
            header = handle.read(32)
        if header.startswith(b"%PDF-"):
            return "pdf"
        if header.startswith(b"\x89PNG\r\n\x1a\n") or header.startswith((b"\xff\xd8\xff", b"GIF87a", b"GIF89a")):
            return "image"
        if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return "image"
        if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
            return "audio"
        if header.startswith(b"ID3"):
            return "audio"
        if len(header) >= 12 and header[4:8] == b"ftyp":
            return "video"
    except OSError:
        return "unknown"

    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
            if "[Content_Types].xml" in names and "xl/workbook.xml" in names:
                return "xlsx"
            return "zip"
    except (OSError, zipfile.BadZipFile):
        pass
    return suffix.lstrip(".") or "unknown"
