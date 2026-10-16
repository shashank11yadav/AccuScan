"""Pinned Google Fonts handwriting families (OFL) and the train/held-out split."""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .download import ChecksumError, fetch_to, sha256_file
from .paths import CONFIG_DIR

FONTS_CONFIG = CONFIG_DIR / "fonts.toml"
OFL_MARKERS = ("SIL OPEN FONT LICENSE", "VERSION 1.1")


@dataclass(frozen=True)
class FontSpec:
    family_dir: str
    file: str
    sha256: str
    license_sha256: str
    heldout: bool

    def url(self, repo: str, commit: str, name: str) -> str:
        return f"https://raw.githubusercontent.com/{repo}/{commit}/ofl/{self.family_dir}/{name}"

    def font_path(self, root: Path) -> Path:
        return root / self.family_dir / self.file

    def license_path(self, root: Path) -> Path:
        return root / self.family_dir / "OFL.txt"


@dataclass(frozen=True)
class FontSet:
    repo: str
    commit: str
    fonts: tuple[FontSpec, ...]

    @property
    def train(self) -> tuple[FontSpec, ...]:
        return tuple(f for f in self.fonts if not f.heldout)

    @property
    def heldout(self) -> tuple[FontSpec, ...]:
        return tuple(f for f in self.fonts if f.heldout)


def load_font_set(path: Path = FONTS_CONFIG) -> FontSet:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    heldout = set(raw["heldout"])
    fonts = tuple(
        FontSpec(f["dir"], f["file"], f["sha256"], f["license_sha256"], f["dir"] in heldout)
        for f in raw["font"]
    )
    unknown = heldout - {f.family_dir for f in fonts}
    if unknown:
        raise ValueError(f"held-out fonts not in the list: {sorted(unknown)}")
    return FontSet(raw["repo"], raw["commit"], fonts)


def is_ofl(text: str) -> bool:
    upper = " ".join(text.upper().split())
    return all(marker in upper for marker in OFL_MARKERS)


def verify_font(spec: FontSpec, root: Path) -> None:
    """Raise unless both files are present, match their pins, and the licence is OFL 1.1."""
    for path, expected in (
        (spec.font_path(root), spec.sha256),
        (spec.license_path(root), spec.license_sha256),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run `accuscan fonts`")
        if sha256_file(path) != expected:
            raise ChecksumError(f"{path} does not match its pinned SHA-256")
    if not is_ofl(spec.license_path(root).read_text(encoding="utf-8", errors="replace")):
        raise ValueError(f"{spec.license_path(root)} is not the SIL Open Font License 1.1")


def download_fonts(
    font_set: FontSet, root: Path, log: Callable[[str], None] = print
) -> list[FontSpec]:
    downloaded = 0
    for spec in font_set.fonts:
        for name, sha in ((spec.file, spec.sha256), ("OFL.txt", spec.license_sha256)):
            dest = root / spec.family_dir / name
            if fetch_to(spec.url(font_set.repo, font_set.commit, name), sha, dest):
                downloaded += 1
        verify_font(spec, root)
    log(f"{len(font_set.fonts)} families verified in {root} ({downloaded} files downloaded)")
    return list(font_set.fonts)


def available_fonts(specs: tuple[FontSpec, ...], root: Path) -> list[Path]:
    """Verified font files for these specs (raises if any is missing or altered)."""
    for spec in specs:
        verify_font(spec, root)
    return [spec.font_path(root) for spec in specs]
