import numpy as np
import pytest

from accuscan.preprocess import (
    INPUT_HEIGHT,
    MAX_WIDTH,
    EmptyImageError,
    preprocess,
    rgba_to_gray,
)


def _word(h: int = 120, w: int = 300, ink: int = 20, paper: int = 235) -> np.ndarray:
    img = np.full((h, w), paper, dtype=np.uint8)
    img[40:80, 30:60] = ink  # a block "letter"
    img[50:70, 90:250] = ink  # a bar
    return img


def test_shape_is_one_by_64_by_multiple_of_four() -> None:
    out = preprocess(_word())
    assert out.dtype == np.float32
    assert out.shape[0] == 1
    assert out.shape[1] == INPUT_HEIGHT
    assert out.shape[2] % 4 == 0
    assert out.shape[2] <= MAX_WIDTH


def test_background_is_minus_one_and_ink_is_plus_one() -> None:
    out = preprocess(_word())
    assert out.min() >= -1.0
    assert out.max() <= 1.0
    assert out[0, 0, 0] == pytest.approx(-1.0)  # padding corner is paper
    assert out.max() == pytest.approx(1.0)


def test_crop_ignores_position_on_the_page() -> None:
    a = np.full((300, 600), 240, dtype=np.uint8)
    b = a.copy()
    a[20:60, 10:200] = 10
    b[200:240, 350:540] = 10
    np.testing.assert_allclose(preprocess(a), preprocess(b), atol=1e-6)


def test_light_ink_on_dark_background_is_inverted() -> None:
    dark = _word(ink=230, paper=25)
    light = _word(ink=25, paper=230)
    np.testing.assert_allclose(preprocess(dark), preprocess(light), atol=1e-6)


def test_contrast_is_normalised_for_grey_paper() -> None:
    grey = _word(ink=90, paper=170)
    crisp = _word(ink=0, paper=255)
    np.testing.assert_allclose(preprocess(grey), preprocess(crisp), atol=1e-6)


def test_aspect_ratio_is_kept() -> None:
    img = np.full((100, 400), 255, dtype=np.uint8)
    img[10:42, 10:138] = 0  # ink box 32 x 128, padded crop 40 x 136
    out = preprocess(img)
    expected = round(136 * INPUT_HEIGHT / 40)
    assert out.shape[2] == expected + (-expected % 4)


def test_very_wide_line_is_squeezed_to_max_width() -> None:
    img = np.full((40, 4000), 255, dtype=np.uint8)
    img[10:30, 5:3995] = 0
    out = preprocess(img)
    assert out.shape == (1, INPUT_HEIGHT, MAX_WIDTH)


@pytest.mark.parametrize("value", [0, 128, 255])
def test_blank_image_raises(value: int) -> None:
    with pytest.raises(EmptyImageError):
        preprocess(np.full((64, 64), value, dtype=np.uint8))


def test_faint_noise_counts_as_blank() -> None:
    rng = np.random.default_rng(0)
    img = (240 + rng.integers(-5, 6, size=(80, 200))).astype(np.uint8)
    with pytest.raises(EmptyImageError):
        preprocess(img)


def test_tiny_input_is_upscaled() -> None:
    img = np.full((6, 10), 255, dtype=np.uint8)
    img[2:4, 3:7] = 0
    out = preprocess(img)
    assert out.shape[1] == INPUT_HEIGHT
    assert out.shape[2] >= 4


def test_rgba_composites_on_white_then_converts_to_grey() -> None:
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[0, 0] = (0, 0, 0, 0)  # transparent -> white
    rgba[0, 1] = (0, 0, 0, 255)  # opaque black
    rgba[1, 0] = (255, 0, 0, 255)  # red -> 0.299
    rgba[1, 1] = (0, 0, 0, 128)  # half black on white
    gray = rgba_to_gray(rgba)
    assert gray.dtype == np.uint8
    assert gray.tolist() == [[255, 0], [76, 127]]
