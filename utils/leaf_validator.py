"""
utils/leaf_validator.py
========================
MaizeLeafValidator — THREE-stage gate that runs BEFORE (and around) the
disease CNN, so that only genuine maize leaf photos ever reach a disease
diagnosis.

Stage 0: General content gate (MobileNetV2 / ImageNet, see
    utils/general_image_gate.py). Runs a general-purpose, pretrained
    image classifier first and rejects the image immediately if it is a
    confident match for a known animal, a person-indicator (clothing,
    accessories, grooming items), or a common man-made object (vehicles,
    furniture, electronics, buildings, household items). This is the
    main line of defence against photos of people, animals, and things.

Stage 1: Visual feature analysis (no model needed, runs in <5ms)
    Checks 6 visual signatures that maize leaves always have:
    - Green channel dominance (leaves are green)
    - Aspect ratio of the image content
    - Saturation levels (leaves have moderate saturation)
    - Not a blank/white/black image
    - Minimum texture variance (leaves have texture, not solid fills)
    - RGB balance (not a photo of sky, soil, paper, skin)

Stage 2: CNN entropy gate (runs after Stage 1 passes)
    After the disease model produces softmax probs, checks:
    - If max probability < MIN_CONF_THRESHOLD the model is completely
      uncertain → the image is likely not a maize leaf at all.
    - If Shannon entropy of probs > ENTROPY_THRESHOLD → model sees
      no clear pattern → likely not a maize leaf.

Together, these three stages:
    ✔ Reject photos of people, animals, and everyday objects (Stage 0)
    ✔ Reject blank/white/black images (Stage 1)
    ✔ Reject images with very low green content — soil, paper, sky (Stage 1)
    ✔ Reject anything the disease CNN itself has no confident pattern for (Stage 2)
    ✔ Fail OPEN on any internal error in Stage 0 — a missing/unavailable
      general-content model can never crash or fully block the app,
      because Stage 1 and Stage 2 still run regardless
    ✔ Never reject actual maize leaf images that the CNN trained on
      (thresholds are deliberately conservative)

Author: Brice Gaetan Nono Youmbi | ULK Data Science 2025/2026
"""

import logging
import numpy as np
from PIL import Image as PILImage

from utils.general_image_gate import check_general_content

log = logging.getLogger("MaizeLeafValidator")


class ValidationResult:
    """
    Result of MaizeLeafValidator.validate().

    Attributes:
        is_valid    (bool) : True = maize leaf, False = rejected
        reason_code (str)  : machine-readable rejection code
        reason_text (str)  : human-readable rejection message shown to user
        details     (dict) : numeric feature values for debugging/logging
    """

    # Rejection codes
    CODE_OK               = "ok"
    CODE_DETECTED_ANIMAL  = "detected_animal"
    CODE_DETECTED_PERSON  = "detected_person"
    CODE_DETECTED_OBJECT  = "detected_object"
    CODE_NOT_GREEN        = "not_green"
    CODE_BLANK            = "blank_image"
    CODE_LOW_TEXTURE      = "low_texture"
    CODE_WRONG_COLOR_DIST = "wrong_color_distribution"
    CODE_LOW_SATURATION   = "low_saturation"
    CODE_CNN_UNCERTAIN    = "cnn_uncertain"

    USER_MESSAGE = (
        "Sorry, the system cannot analyse your image. "
        "Please upload a real maize leaf photo."
    )

    # Per-code extra hints shown below the main message
    HINTS = {
        "detected_animal": (
            "This looks like a photo of an animal, not a maize leaf. "
            "MaizeScan only diagnoses maize (corn) leaves."
        ),
        "detected_person": (
            "This looks like a photo of a person (or clothing/accessories), "
            "not a maize leaf. MaizeScan only diagnoses maize (corn) leaves."
        ),
        "detected_object": (
            "This looks like a photo of an everyday object, not a maize leaf. "
            "MaizeScan only diagnoses maize (corn) leaves."
        ),
        "not_green": (
            "The image does not appear to contain a green plant leaf. "
            "Make sure the leaf fills most of the frame."
        ),
        "blank_image": (
            "The image appears blank, solid-coloured, or has very low content. "
            "Please use a clear photo of a maize leaf."
        ),
        "low_texture": (
            "The image has almost no visual texture. "
            "A clear, focused photo of a leaf surface is needed."
        ),
        "wrong_color_distribution": (
            "The image colours do not match a maize leaf. "
            "Avoid photos of soil, sky, paper, or other plants."
        ),
        "low_saturation": (
            "The image appears washed-out or greyscale. "
            "Use a colour photo of a maize leaf in good lighting."
        ),
        "cnn_uncertain": (
            "The model cannot recognise any maize leaf disease pattern "
            "in this image. Please use a clear, close-up photo of a "
            "single maize leaf."
        ),
    }

    def __init__(self, is_valid: bool, reason_code: str = "ok", details: dict = None):
        self.is_valid    = is_valid
        self.reason_code = reason_code
        self.reason_text = self.USER_MESSAGE
        self.hint        = self.HINTS.get(reason_code, "")
        self.details     = details or {}

    @classmethod
    def ok(cls, details: dict = None) -> "ValidationResult":
        return cls(True, cls.CODE_OK, details)

    @classmethod
    def reject(cls, code: str, details: dict = None) -> "ValidationResult":
        return cls(False, code, details)


class MaizeLeafValidator:
    """
    Two-stage maize leaf validator.

    Usage:
        validator = MaizeLeafValidator()

        # Stage 0 — general content gate (person / animal / object)
        result = validator.validate_general(leaf_image)
        if not result.is_valid:
            return show_rejection_page(result)

        # Stage 1 — visual check (before CNN inference)
        result = validator.validate_visual(leaf_image)
        if not result.is_valid:
            return show_rejection_page(result)

        # Run CNN inference
        probs = cnn.predict(leaf_image.img_array)

        # Stage 2 — CNN entropy check (after inference)
        result = validator.validate_cnn(probs)
        if not result.is_valid:
            return show_rejection_page(result)

    Thresholds are conservative to avoid false rejections of
    diseased leaves (which can be brown, tan, or grey).
    """

    # ── Stage 1 thresholds ────────────────────────────────────────
    # Green ratio: fraction of pixels where G > R and G > B
    # Healthy leaves: ~0.55–0.85. Diseased: ~0.20–0.65 (rust=brown, blight=tan)
    # Non-leaf: usually < 0.10 (skin, paper) or > 0.95 (solid green fill)
    MIN_GREEN_RATIO    = 0.08   # very permissive — diseased leaves can be mostly brown
    MAX_GREEN_RATIO    = 0.98   # rejects solid green fills / non-photo images

    # Mean green channel value (0–255). Leaves: 90–200. Pure white: 240+. Black: <15.
    MIN_MEAN_GREEN     = 25.0   # rejects almost-black images
    MAX_MEAN_GREEN     = 240.0  # rejects almost-white images (paper, blank)

    # Texture variance: std dev of greyscale image. Leaves: 20–80. Solid: <5.
    MIN_TEXTURE_STD    = 8.0    # rejects solid-colour images

    # Saturation (HSV S channel, 0–255). Leaves: 50–220. Greyscale photos: <30.
    MIN_SATURATION     = 18.0   # rejects desaturated / greyscale images

    # Red-minus-blue difference in pixels. Skin: R >> B. Sky: B >> R.
    # Maize leaves span a wide range so we only check extremes.
    MAX_MEAN_RED_MINUS_BLUE  =  90.0   # rejects very red images (skin)
    MIN_MEAN_RED_MINUS_BLUE  = -80.0   # rejects very blue images (sky, water)

    # ── Stage 2 thresholds ────────────────────────────────────────
    # If the CNN's maximum softmax probability is below this, it has
    # no confidence in any class → probably not a maize leaf.
    MIN_CONF_THRESHOLD = 0.30   # very permissive; real leaves usually ≥ 0.50

    # Shannon entropy of softmax probs. Uniform distribution (4 classes) = 1.386.
    # If entropy ≥ MAX_ENTROPY the model sees no pattern at all.
    MAX_ENTROPY        = 1.35   # just below uniform (1.386) — very permissive

    def __init__(self):
        self._log = logging.getLogger("MaizeLeafValidator")

    def validate_general(self, leaf_image) -> "ValidationResult":
        """
        Stage 0: General-purpose content gate. Runs a pretrained
        MobileNetV2 (ImageNet) classifier and rejects the image outright
        if it is a confident match for a known animal, person-indicator,
        or man-made object — before any leaf-specific logic even runs.

        Args:
            leaf_image: LeafImage object (with ._pil_img set)

        Returns:
            ValidationResult — .is_valid=False if a person/animal/object
            was confidently detected; .is_valid=True otherwise (including
            when the general-content model is unavailable, since this
            stage always fails open).
        """
        try:
            pil_img = leaf_image._pil_img
            if pil_img is None:
                # No image to check — let Stage 1 handle this case.
                return ValidationResult.ok({"reason": "no pil_img, skipped stage0"})

            gate_result = check_general_content(pil_img)
            if not gate_result.is_valid:
                self._log.info("REJECT %s | %s", gate_result.reason_code, gate_result.details)
                return ValidationResult.reject(gate_result.reason_code, gate_result.details)

            self._log.debug("PASS stage0 | %s", gate_result.details)
            return ValidationResult.ok(gate_result.details)

        except Exception as exc:
            # Never crash the app due to Stage 0 — Stage 1 & 2 remain
            # the safety net if the general-content gate misbehaves.
            self._log.error("Stage0 error (passing through): %s", exc)
            return ValidationResult.ok({"error": str(exc)})

    def validate_visual(self, leaf_image) -> "ValidationResult":
        """
        Stage 1: Fast visual feature analysis of the preprocessed image.

        Args:
            leaf_image: LeafImage object (with ._pil_img set)

        Returns:
            ValidationResult — .is_valid=True if image passes all checks
        """
        try:
            pil_img = leaf_image._pil_img
            if pil_img is None:
                return ValidationResult.reject(
                    ValidationResult.CODE_BLANK,
                    {"reason": "no pil_img"}
                )

            # Work with a small 64×64 thumbnail for speed
            thumb     = pil_img.convert("RGB").resize((64, 64), PILImage.LANCZOS)
            arr       = np.array(thumb, dtype=np.float32)   # (64, 64, 3)
            R, G, B   = arr[:,:,0], arr[:,:,1], arr[:,:,2]

            # ── Check 1: not blank ────────────────────────────────
            mean_green = float(G.mean())
            if mean_green < self.MIN_MEAN_GREEN or mean_green > self.MAX_MEAN_GREEN:
                details = {"mean_green": round(mean_green, 2)}
                self._log.info("REJECT blank | %s", details)
                return ValidationResult.reject(ValidationResult.CODE_BLANK, details)

            # ── Check 2: texture variance ──────────────────────────
            grey    = 0.299*R + 0.587*G + 0.114*B
            tex_std = float(grey.std())
            if tex_std < self.MIN_TEXTURE_STD:
                details = {"texture_std": round(tex_std, 2)}
                self._log.info("REJECT low_texture | %s", details)
                return ValidationResult.reject(ValidationResult.CODE_LOW_TEXTURE, details)

            # ── Check 3: green dominance ───────────────────────────
            green_mask  = (G > R) & (G > B)
            green_ratio = float(green_mask.mean())
            # Also count pixels where green is close to dominant
            # (diseased leaves often have tan/brown areas where R≈G)
            soft_green  = (G > B) & (G > 20)
            soft_ratio  = float(soft_green.mean())
            effective_green = max(green_ratio, soft_ratio * 0.5)
            if effective_green < self.MIN_GREEN_RATIO:
                details = {"green_ratio": round(green_ratio,3),
                           "soft_ratio": round(soft_ratio,3)}
                self._log.info("REJECT not_green | %s", details)
                return ValidationResult.reject(ValidationResult.CODE_NOT_GREEN, details)

            # ── Check 4: saturation ────────────────────────────────
            hsv_arr    = np.array(thumb.convert("HSV"), dtype=np.float32)
            mean_sat   = float(hsv_arr[:,:,1].mean())
            if mean_sat < self.MIN_SATURATION:
                details = {"mean_saturation": round(mean_sat, 2)}
                self._log.info("REJECT low_saturation | %s", details)
                return ValidationResult.reject(ValidationResult.CODE_LOW_SATURATION, details)

            # ── Check 5: colour distribution extremes ─────────────
            mean_r_minus_b = float((R - B).mean())
            if (mean_r_minus_b > self.MAX_MEAN_RED_MINUS_BLUE or
                    mean_r_minus_b < self.MIN_MEAN_RED_MINUS_BLUE):
                details = {"mean_r_minus_b": round(mean_r_minus_b, 2)}
                self._log.info("REJECT wrong_color_distribution | %s", details)
                return ValidationResult.reject(
                    ValidationResult.CODE_WRONG_COLOR_DIST, details
                )

            # ── All visual checks passed ───────────────────────────
            details = {
                "mean_green":    round(mean_green, 2),
                "green_ratio":   round(green_ratio, 3),
                "texture_std":   round(tex_std, 2),
                "mean_sat":      round(mean_sat, 2),
                "r_minus_b":     round(mean_r_minus_b, 2),
            }
            self._log.debug("PASS visual | %s", details)
            return ValidationResult.ok(details)

        except Exception as exc:
            # Never crash the app due to validation — log and pass through
            self._log.error("Validation error (passing through): %s", exc)
            return ValidationResult.ok({"error": str(exc)})

    def validate_cnn(self, probs: np.ndarray) -> "ValidationResult":
        """
        Stage 2: Check CNN output entropy and max confidence.

        Args:
            probs: np.ndarray shape (4,) — softmax output from CNNModel.predict()

        Returns:
            ValidationResult — .is_valid=True if model recognised a pattern
        """
        try:
            probs    = np.array(probs, dtype=float)
            max_prob = float(probs.max())
            # Shannon entropy: H = -sum(p * log(p))
            safe_p  = np.clip(probs, 1e-9, 1.0)
            entropy = float(-np.sum(safe_p * np.log(safe_p)))

            details = {
                "max_prob": round(max_prob, 4),
                "entropy":  round(entropy, 4),
            }

            if max_prob < self.MIN_CONF_THRESHOLD:
                self._log.info("REJECT cnn_uncertain (max_prob too low) | %s", details)
                return ValidationResult.reject(
                    ValidationResult.CODE_CNN_UNCERTAIN, details
                )

            if entropy > self.MAX_ENTROPY:
                self._log.info("REJECT cnn_uncertain (entropy too high) | %s", details)
                return ValidationResult.reject(
                    ValidationResult.CODE_CNN_UNCERTAIN, details
                )

            self._log.debug("PASS cnn | %s", details)
            return ValidationResult.ok(details)

        except Exception as exc:
            self._log.error("CNN validation error (passing through): %s", exc)
            return ValidationResult.ok({"error": str(exc)})
