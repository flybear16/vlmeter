"""Image loading, optional downscaling, and base64 encoding for each provider."""
from __future__ import annotations

import base64
import io
from pathlib import Path

MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# Providers differ; keep requests lean by capping the longest side.
MAX_SIDE = 1568
MAX_BYTES = 4 * 1024 * 1024


def _load_bytes(path: str | Path) -> tuple[bytes, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    mime = MIME_BY_EXT.get(p.suffix.lower(), "image/png")
    return p.read_bytes(), mime


def prepare_image(path: str | Path) -> tuple[bytes, str]:
    """Return (raw_bytes, mime). Downscale/re-encode oversized images."""
    raw, mime = _load_bytes(path)
    if len(raw) <= MAX_BYTES:
        return raw, mime

    from PIL import Image  # imported lazily so non-image runs need no Pillow

    img = Image.open(io.BytesIO(raw))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def b64_image(path: str | Path) -> tuple[str, str]:
    """Return (base64_no_prefix, mime)."""
    raw, mime = prepare_image(path)
    return base64.b64encode(raw).decode("ascii"), mime


def data_url(path: str | Path) -> str:
    """Return a data: URL, accepted by OpenAI-compatible chat endpoints."""
    b64, mime = b64_image(path)
    return f"data:{mime};base64,{b64}"
