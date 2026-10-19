import hashlib
import io
import urllib.request
from pathlib import Path
from types import TracebackType

import pytest

from accuscan import download
from accuscan.download import ChecksumError, fetch, fetch_to

URL = "https://example.org/file.txt"


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def serve(monkeypatch: pytest.MonkeyPatch, body: bytes) -> list[str]:
    requested: list[str] = []

    def urlopen(request: object, timeout: float) -> FakeResponse:
        requested.append(getattr(request, "full_url", ""))
        return FakeResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return requested


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_fetch_refuses_plain_http(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = serve(monkeypatch, b"x")
    with pytest.raises(ValueError, match="non-HTTPS"):
        fetch("http://example.org/file.txt", sha(b"x"))
    assert requested == []


def test_fetch_returns_bytes_that_match_the_pinned_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(monkeypatch, b"hello")
    assert fetch(URL, sha(b"hello")) == b"hello"


def test_fetch_rejects_a_hash_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(monkeypatch, b"tampered")
    with pytest.raises(ChecksumError):
        fetch(URL, sha(b"hello"))


def test_fetch_stops_reading_past_the_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(download, "MAX_DOWNLOAD_BYTES", 8)
    serve(monkeypatch, b"0123456789")
    with pytest.raises(ValueError, match="larger than 8 bytes"):
        fetch(URL, sha(b"0123456789"))


def test_fetch_to_writes_nothing_when_the_hash_is_wrong(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    serve(monkeypatch, b"tampered")
    dest = tmp_path / "sub" / "file.txt"
    with pytest.raises(ChecksumError):
        fetch_to(URL, sha(b"hello"), dest)
    assert not dest.exists()
    assert list(tmp_path.rglob("*")) == []


def test_fetch_to_replaces_the_file_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    serve(monkeypatch, b"hello")
    dest = tmp_path / "file.txt"
    dest.write_bytes(b"stale")
    assert fetch_to(URL, sha(b"hello"), dest)
    assert dest.read_bytes() == b"hello"
    assert [p.name for p in tmp_path.iterdir()] == ["file.txt"]
    assert not fetch_to(URL, sha(b"hello"), dest)  # already there: no download
