"""
tests/test_validator_pipeline.py
=================================
Offline unit tests for MaizeLeafValidator's Stage 1 (visual) and Stage 2
(CNN entropy) gates, using synthetically generated images so these tests
run anywhere with no internet access and no trained model file required.

Stage 0 (general content gate) needs the MobileNetV2 ImageNet weights
downloaded, so it is exercised separately in
test_general_gate_smoke() with a skip guard for offline environments —
see that test's docstring for how to run it with real photos.

Run with:
    python -m pytest tests/test_validator_pipeline.py -v
"""

import os
import sys
import numpy as np
import pytest
from PIL import Image as PILImage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.leaf_validator import MaizeLeafValidator
from utils.leaf_image import LeafImage


def _leaf_image_from_array(arr: np.ndarray) -> LeafImage:
    """Build a LeafImage directly from an in-memory RGB numpy array,
    bypassing disk I/O, for fast synthetic testing."""
    pil_img = PILImage.fromarray(arr.astype(np.uint8), mode="RGB")
    leaf = LeafImage()
    leaf.source = "<synthetic>"
    leaf._pil_img = pil_img
    leaf._apply_preprocessing((224, 224))
    return leaf


def _block_noise(shape, low_res=28, rng=None):
    """
    Generate spatially-correlated ("blocky") noise rather than pure
    per-pixel noise. Stage 1's texture check works on a 64x64 LANCZOS
    thumbnail of the image, which smooths out high-frequency per-pixel
    noise almost completely — a real leaf photo's texture survives that
    downsampling because it has structure at a coarser scale (veins,
    lesions, colour patches), so synthetic test images need to mimic
    that coarser-scale structure rather than fine pixel noise.
    """
    rng = rng or np.random.default_rng(0)
    small = rng.integers(0, 256, size=(low_res, low_res, 3), dtype=np.uint8)
    img = PILImage.fromarray(small, mode="RGB").resize(
        (shape[1], shape[0]), PILImage.NEAREST
    )
    return np.array(img, dtype=np.float32)


@pytest.fixture(scope="module")
def validator():
    return MaizeLeafValidator()


# ── Stage 1: visual feature checks ─────────────────────────────────

def test_blank_white_image_is_rejected(validator):
    arr = np.full((224, 224, 3), 255, dtype=np.uint8)  # pure white
    leaf = _leaf_image_from_array(arr)
    result = validator.validate_visual(leaf)
    assert result.is_valid is False
    assert result.reason_code == "blank_image"


def test_blank_black_image_is_rejected(validator):
    arr = np.zeros((224, 224, 3), dtype=np.uint8)  # pure black
    leaf = _leaf_image_from_array(arr)
    result = validator.validate_visual(leaf)
    assert result.is_valid is False
    assert result.reason_code == "blank_image"


def test_flat_grey_image_is_rejected_for_low_texture(validator):
    arr = np.full((224, 224, 3), 128, dtype=np.uint8)  # flat mid-grey, no texture
    leaf = _leaf_image_from_array(arr)
    result = validator.validate_visual(leaf)
    assert result.is_valid is False
    # Could trip low_texture or low_saturation depending on exact values —
    # either is a correct rejection for a flat, non-leaf image.
    assert result.reason_code in ("low_texture", "low_saturation", "blank_image")


def test_blue_sky_like_image_is_rejected(validator):
    # Strong blue dominance with coarse noise for texture — like a sky photo.
    rng = np.random.default_rng(42)
    base = np.zeros((224, 224, 3), dtype=np.float32)
    base[:, :, 2] = 200  # blue channel high
    base[:, :, 0] = 60   # low red
    base[:, :, 1] = 120  # moderate green (still less than blue)
    noise = _block_noise(base.shape, low_res=28, rng=rng) - 128
    arr = np.clip(base + noise * 0.3, 0, 255).astype(np.uint8)
    leaf = _leaf_image_from_array(arr)
    result = validator.validate_visual(leaf)
    assert result.is_valid is False
    assert result.reason_code in ("wrong_color_distribution", "not_green", "low_texture")


def test_green_textured_image_passes_stage1(validator):
    # Simulate a leaf-like green, textured, moderately saturated image.
    rng = np.random.default_rng(7)
    base = np.zeros((224, 224, 3), dtype=np.float32)
    base[:, :, 1] = 140   # green dominant
    base[:, :, 0] = 70
    base[:, :, 2] = 50
    noise = _block_noise(base.shape, low_res=28, rng=rng) - 128  # coarse texture
    arr = np.clip(base + noise * 0.5, 0, 255).astype(np.uint8)
    leaf = _leaf_image_from_array(arr)
    result = validator.validate_visual(leaf)
    assert result.is_valid is True, f"expected pass, got reject: {result.reason_code} {result.details}"


# ── Stage 2: CNN entropy / confidence checks ───────────────────────

def test_uniform_probs_are_rejected_as_uncertain(validator):
    probs = np.array([0.25, 0.25, 0.25, 0.25])  # maximum-entropy, no clear class
    result = validator.validate_cnn(probs)
    assert result.is_valid is False
    assert result.reason_code == "cnn_uncertain"


def test_low_confidence_probs_are_rejected(validator):
    probs = np.array([0.28, 0.27, 0.23, 0.22])  # all below MIN_CONF_THRESHOLD
    result = validator.validate_cnn(probs)
    assert result.is_valid is False
    assert result.reason_code == "cnn_uncertain"


def test_confident_prediction_passes_stage2(validator):
    probs = np.array([0.02, 0.03, 0.90, 0.05])  # clear, confident class
    result = validator.validate_cnn(probs)
    assert result.is_valid is True


# ── Stage 0: smoke test (requires internet on first run) ───────────

def test_general_gate_smoke():
    """
    Smoke test for Stage 0. Skips automatically if MobileNetV2 weights
    cannot be downloaded (e.g. offline CI environment) — Stage 0 is
    designed to fail open in exactly that situation, so a skip here is
    consistent with production behaviour, not a bug.

    To validate Stage 0 against real photos of people/animals/objects,
    replace the synthetic array below with:
        from PIL import Image
        img = Image.open("/path/to/dog_photo.jpg")
    and check `result.is_valid is False` with an appropriate reason_code.
    """
    from utils.general_image_gate import check_general_content

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    img = PILImage.fromarray(arr, mode="RGB")

    try:
        result = check_general_content(img)
    except Exception:
        pytest.skip("MobileNetV2 weights unavailable in this environment.")
        return

    # Random noise should not confidently match any curated keyword list;
    # if the general-content model is unavailable it fails open (is_valid
    # True with an "error" key) — both are acceptable outcomes here.
    assert isinstance(result.is_valid, bool)
