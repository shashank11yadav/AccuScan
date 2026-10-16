from pathlib import Path

import pytest

from accuscan.download import ChecksumError, sha256_bytes
from accuscan.fonts import FontSpec, is_ofl, load_font_set, verify_font

OFL_TEXT = (
    "Copyright 2020\n\nThis Font Software is licensed under the SIL Open Font License, "
    "Version 1.1.\n"
)


def test_pinned_list_has_40_to_80_families_with_a_15_percent_holdout() -> None:
    fonts = load_font_set()
    assert 40 <= len(fonts.fonts) <= 80
    assert len({f.family_dir for f in fonts.fonts}) == len(fonts.fonts)
    share = len(fonts.heldout) / len(fonts.fonts)
    assert 0.12 <= share <= 0.18
    assert not {f.family_dir for f in fonts.train} & {f.family_dir for f in fonts.heldout}
    assert len(fonts.commit) == 40
    for spec in fonts.fonts:
        assert len(spec.sha256) == 64
        assert len(spec.license_sha256) == 64
        assert spec.file.endswith(".ttf")


def test_urls_point_at_the_ofl_folder_of_the_pinned_commit() -> None:
    fonts = load_font_set()
    spec = fonts.fonts[0]
    url = spec.url(fonts.repo, fonts.commit, spec.file)
    assert url.startswith(f"https://raw.githubusercontent.com/google/fonts/{fonts.commit}/ofl/")


def test_is_ofl_accepts_the_licence_and_rejects_others() -> None:
    assert is_ofl(OFL_TEXT)
    assert is_ofl("SIL OPEN FONT LICENSE\nVersion 1.1 - 26 February 2007")
    assert not is_ofl("Licensed under the Apache License, Version 2.0")


def _write(root: Path, font: bytes, licence: str) -> FontSpec:
    (root / "fam").mkdir()
    (root / "fam" / "Fam.ttf").write_bytes(font)
    (root / "fam" / "OFL.txt").write_text(licence)
    return FontSpec("fam", "Fam.ttf", sha256_bytes(font), sha256_bytes(licence.encode()), False)


def test_verify_font_accepts_matching_files(tmp_path: Path) -> None:
    verify_font(_write(tmp_path, b"font", OFL_TEXT), tmp_path)


def test_verify_font_rejects_a_changed_file(tmp_path: Path) -> None:
    spec = _write(tmp_path, b"font", OFL_TEXT)
    (tmp_path / "fam" / "Fam.ttf").write_bytes(b"tampered")
    with pytest.raises(ChecksumError):
        verify_font(spec, tmp_path)


def test_verify_font_rejects_a_non_ofl_licence(tmp_path: Path) -> None:
    spec = _write(tmp_path, b"font", "Apache License 2.0")
    with pytest.raises(ValueError, match="Open Font License"):
        verify_font(spec, tmp_path)


def test_verify_font_reports_missing_files(tmp_path: Path) -> None:
    spec = FontSpec("nope", "Nope.ttf", "0" * 64, "0" * 64, False)
    with pytest.raises(FileNotFoundError):
        verify_font(spec, tmp_path)
