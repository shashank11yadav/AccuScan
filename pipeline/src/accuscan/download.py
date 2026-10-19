"""HTTPS downloads checked against a pinned SHA-256."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

USER_AGENT = "accuscan-pipeline/2.0"
TIMEOUT_S = 60
# The largest pinned file is a few megabytes; anything far bigger is not it.
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024


class ChecksumError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, sha256: str) -> bytes:
    """Download url (https only) and verify its SHA-256 before returning it."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-HTTPS URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310
        data: bytes = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"{url}: larger than {MAX_DOWNLOAD_BYTES} bytes")
    actual = sha256_bytes(data)
    if actual != sha256:
        raise ChecksumError(f"{url}: expected sha256 {sha256}, got {actual}")
    return data


def fetch_to(url: str, sha256: str, dest: Path) -> bool:
    """Download to dest unless a file with the right hash is already there.

    Returns True if a download happened. Writes to a temporary name first so an
    interrupted run never leaves a partial file under the final name.
    """
    if dest.exists() and sha256_file(dest) == sha256:
        return False
    data = fetch(url, sha256)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return True
