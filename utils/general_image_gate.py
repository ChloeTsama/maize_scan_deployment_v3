"""
utils/general_image_gate.py
============================
Stage 0 — General-purpose "is this even a plant?" content gate.

WHY THIS EXISTS
----------------
The original MaizeLeafValidator (Stage 1 + Stage 2) relies on colour and
texture heuristics plus the disease CNN's own confidence/entropy. Those
checks are fast and catch most blank/plain/wrong-colour images, but a
green, textured object that is NOT a maize leaf (e.g. a person in a green
shirt against grass, a green plastic toy, a parrot, a lawn, another
plant's leaf) can still slip past pure colour/texture rules.

This module adds a general-purpose, pretrained image classifier
(MobileNetV2, trained on the 1000-class ImageNet benchmark) as an extra
gate that runs BEFORE the maize disease CNN. ImageNet was not built for
maize disease detection, but it already knows what hundreds of common
animals, people-related objects (clothing, accessories), vehicles,
furniture, and household items look like. We use it purely as a coarse
"is this obviously a person / animal / man-made object?" filter — not as
a leaf classifier.

Decision rule (deliberately conservative, to avoid rejecting real leaves):
    - If MobileNetV2's top prediction is a confident match (prob >= 0.15)
      against a curated list of animal / person-indicator / man-made-object
      ImageNet labels -> REJECT immediately with a specific reason.
    - Otherwise -> PASS THROUGH to Stage 1 (colour/texture) and Stage 2
      (CNN entropy). We do NOT require a positive "this is a plant" match,
      because ImageNet has no "maize leaf" class and forcing a positive
      match would create false rejections of real, valid leaf photos.

The model is loaded lazily (first request only) and cached in memory for
the lifetime of the process, so it adds no cost to app startup and only
adds latency to the very first prediction request.

Author: Brice Gaetan Nono Youmbi | ULK Data Science 2025/2026
"""

import logging
import numpy as np

log = logging.getLogger("GeneralImageGate")

# ── Curated ImageNet label keyword lists ──────────────────────────────
# Labels from keras.applications MobileNetV2 decode_predictions() come
# back as lowercase, underscore-separated strings, e.g. "golden_retriever",
# "sunglasses", "sports_car". We match by substring so that e.g. "terrier"
# catches every one of the ~25 terrier breeds in ImageNet without listing
# them all individually.

ANIMAL_KEYWORDS = [
    # Mammals
    "dog", "terrier", "retriever", "spaniel", "hound", "poodle", "pug",
    "bulldog", "collie", "shepherd", "husky", "corgi", "chihuahua",
    "cat", "tabby", "siamese", "persian", "lynx", "cougar", "leopard",
    "cheetah", "lion", "tiger", "bear", "panda", "elephant", "horse",
    "zebra", "cow", "ox", "bison", "buffalo", "sheep", "ram", "goat",
    "pig", "hog", "boar", "camel", "llama", "deer", "gazelle", "impala",
    "antelope", "fox", "wolf", "hyena", "jackal", "raccoon", "skunk",
    "squirrel", "beaver", "otter", "mouse", "rat", "hamster", "rabbit",
    "hare", "kangaroo", "koala", "monkey", "chimp", "gorilla", "orangutan",
    "baboon", "macaque", "gibbon", "mole", "hedgehog", "armadillo",
    "sloth", "porcupine", "weasel", "mink", "polecat", "marmot",
    # Birds
    "bird", "hen", "cock", "rooster", "chicken", "duck", "goose", "swan",
    "owl", "eagle", "hawk", "kite", "vulture", "falcon", "parrot", "macaw",
    "cockatoo", "peacock", "flamingo", "penguin", "ostrich", "stork",
    "crane", "heron", "pelican", "toucan", "hornbill", "jay", "magpie",
    "crow", "raven", "sparrow", "finch", "robin", "hummingbird",
    "woodpecker", "kingfisher", "quail", "partridge", "grouse",
    # Fish / aquatic
    "fish", "shark", "ray", "eel", "salmon", "trout", "goldfish", "gar",
    "sturgeon", "puffer", "whale", "dolphin", "seal", "sea_lion", "otter",
    "turtle", "tortoise", "terrapin", "crocodile", "alligator", "lizard",
    "gecko", "iguana", "chameleon", "komodo", "snake", "cobra", "viper",
    "python", "boa", "mamba", "frog", "toad", "newt", "salamander",
    "axolotl", "starfish", "jellyfish", "sea_anemone", "sea_urchin",
    "coral", "crab", "lobster", "crayfish", "shrimp", "isopod", "conch",
    "snail", "slug", "chiton",
    # Insects / arachnids / other invertebrates
    "spider", "scorpion", "tick", "insect", "beetle", "butterfly", "moth",
    "dragonfly", "damselfly", "cricket", "grasshopper", "cicada",
    "cockroach", "mantis", "ant", "bee", "wasp", "hornet", "fly",
    "ladybug", "centipede", "millipede", "earthworm", "flatworm",
]

PERSON_INDICATOR_KEYWORDS = [
    # ImageNet has no generic "person" class, but confident predictions
    # for clothing / accessories / grooming items strongly suggest a
    # photo of a person rather than a leaf.
    "suit", "necktie", "bow_tie", "windsor_tie", "jersey", "sweatshirt",
    "cardigan", "kimono", "gown", "academic_gown", "bikini", "brassiere",
    "miniskirt", "sunglasses", "sunglass", "cowboy_hat", "sombrero",
    "bonnet", "shower_cap", "bathing_cap", "hair_spray", "lipstick",
    "wig", "diaper", "mask", "ski_mask", "microphone", "hand-held_computer",
    "cellular_telephone", "band_aid", "stethoscope", "military_uniform",
    "bulletproof_vest", "sandal", "running_shoe", "cowboy_boot",
    "backpack", "purse", "handbag", "wallet", "crash_helmet",
    "football_helmet", "gasmask",
]

OBJECT_KEYWORDS = [
    # Vehicles
    "car", "convertible", "sports_car", "cab", "limousine", "jeep",
    "minivan", "pickup", "truck", "trailer", "moped", "motor_scooter",
    "motorcycle", "bicycle", "unicycle", "tricycle", "train", "locomotive",
    "streetcar", "airliner", "airplane", "warplane", "airship", "balloon",
    "boat", "canoe", "kayak", "gondola", "ship", "yacht", "catamaran",
    "submarine", "aircraft_carrier", "forklift", "tractor", "harvester",
    "snowplow", "tank", "half_track", "bobsled", "dogsled", "go-kart",
    # Furniture / household
    "chair", "throne", "sofa", "couch", "table", "desk", "bookcase",
    "wardrobe", "cabinet", "cradle", "crib", "bassinet", "four-poster",
    "bed", "bench", "bathtub", "toilet_seat", "shower_curtain",
    "medicine_chest", "chest", "china_cabinet", "file_cabinet",
    "refrigerator", "washer", "dishwasher", "microwave", "toaster",
    "vacuum", "iron", "sewing_machine", "espresso_maker", "rotisserie",
    # Electronics
    "television", "monitor", "screen", "computer", "laptop", "keyboard",
    "mouse,_computer_mouse", "printer", "scanner", "camera", "projector",
    "radio", "cassette", "tape_player", "cd_player", "loudspeaker",
    "remote_control", "modem", "joystick",
    # Buildings / structures
    "church", "mosque", "palace", "castle", "monastery", "library",
    "restaurant", "barbershop", "bookshop", "butcher_shop", "grocery_store",
    "toyshop", "greenhouse", "boathouse", "barn", "silo", "birdhouse",
    "dam", "bridge", "viaduct", "lighthouse", "stupa", "planetarium",
    "obelisk", "totem_pole", "triumphal_arch", "cliff_dwelling",
    "prison", "fire_screen", "picket_fence", "chainlink_fence",
    "worm_fence", "stone_wall",
    # Common household / everyday objects
    "bottle", "cup", "mug", "coffee_mug", "wine_bottle", "beer_bottle",
    "plate", "bowl", "spoon", "fork", "knife", "vase", "candle",
    "pillow", "quilt", "blanket", "towel", "umbrella", "clock",
    "wall_clock", "analog_clock", "digital_clock", "hourglass",
    "scissors", "screwdriver", "hammer", "wrench", "chainsaw", "shovel",
    "broom", "mop", "ladder", "barrel", "bucket", "basket", "crate",
    "carton", "envelope", "book", "notebook", "binder", "paper_towel",
    "toilet_paper", "soap_dispenser", "lotion", "perfume", "safety_pin",
    "padlock", "key", "combination_lock", "traffic_light", "street_sign",
    "parking_meter", "mailbox", "fire_hydrant", "manhole_cover",
    "shopping_cart", "shopping_basket", "wheelbarrow", "swing",
    "seesaw", "trampoline", "volleyball", "basketball", "soccer_ball",
    "rugby_ball", "baseball", "golf_ball", "tennis_ball", "ping-pong_ball",
    "ball", "balloon", "kite", "frisbee", "dumbbell", "barbell",
    "punching_bag", "horizontal_bar", "parallel_bars", "ski", "snowboard",
    "skateboard", "surfboard", "paddle",
]

# Minimum confidence on the TOP-1 prediction before we trust a
# animal/person/object match enough to reject. Kept fairly low (0.15)
# because ImageNet's 1000 classes rarely produce a single very high-
# confidence score on out-of-distribution inputs, but a genuine photo of
# a dog/car/person will still usually clear this bar easily (often 0.4+).
MIN_MATCH_CONFIDENCE = 0.15

_model = None  # lazy-loaded MobileNetV2 singleton


def _get_model():
    """Load MobileNetV2 (ImageNet weights) once and cache it in memory."""
    global _model
    if _model is None:
        from tensorflow.keras.applications import MobileNetV2
        log.info("Loading MobileNetV2 (ImageNet) for the general content gate...")
        _model = MobileNetV2(weights="imagenet", include_top=True)
        log.info("MobileNetV2 ready.")
    return _model


def _predict_topk(pil_img, k=5):
    """
    Run MobileNetV2 on a PIL image and return the top-k
    (label, description, probability) tuples.
    """
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input, decode_predictions,
    )

    model = _get_model()
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds = model.predict(arr, verbose=0)
    decoded = decode_predictions(preds, top=k)[0]  # [(wnid, label, prob), ...]
    return [(label, float(prob)) for (_wnid, label, prob) in decoded]


class GeneralGateResult:
    def __init__(self, is_valid, reason_code="ok", details=None):
        self.is_valid = is_valid
        self.reason_code = reason_code
        self.details = details or {}

    @classmethod
    def ok(cls, details=None):
        return cls(True, "ok", details)

    @classmethod
    def reject(cls, code, details=None):
        return cls(False, code, details)


def check_general_content(pil_img) -> GeneralGateResult:
    """
    Stage 0 entry point. Runs MobileNetV2 on the image and rejects it
    if the top prediction confidently matches a known animal, person-
    indicator, or man-made-object ImageNet label.

    Never raises — on any internal failure it returns .ok() so a broken
    or unavailable general-gate model can never block real predictions;
    Stage 1 and Stage 2 remain the safety net either way.
    """
    try:
        top5 = _predict_topk(pil_img, k=5)
        top1_label, top1_prob = top5[0]
        label_norm = top1_label.lower()

        details = {"top5": top5}

        if top1_prob >= MIN_MATCH_CONFIDENCE:
            if any(kw in label_norm for kw in ANIMAL_KEYWORDS):
                details["matched"] = ("animal", top1_label, round(top1_prob, 3))
                log.info("Stage0 REJECT animal | %s", details["matched"])
                return GeneralGateResult.reject("detected_animal", details)

            if any(kw in label_norm for kw in PERSON_INDICATOR_KEYWORDS):
                details["matched"] = ("person_indicator", top1_label, round(top1_prob, 3))
                log.info("Stage0 REJECT person-indicator | %s", details["matched"])
                return GeneralGateResult.reject("detected_person", details)

            if any(kw in label_norm for kw in OBJECT_KEYWORDS):
                details["matched"] = ("object", top1_label, round(top1_prob, 3))
                log.info("Stage0 REJECT object | %s", details["matched"])
                return GeneralGateResult.reject("detected_object", details)

        log.debug("Stage0 PASS | top1=%s (%.3f)", top1_label, top1_prob)
        return GeneralGateResult.ok(details)

    except Exception as exc:
        # Fail OPEN: if the general gate itself errors out (e.g. model
        # could not be downloaded on a restricted network), we do not
        # want that to break the whole prediction pipeline. Stage 1 and
        # Stage 2 still run afterwards.
        log.warning("General content gate unavailable, passing through: %s", exc)
        return GeneralGateResult.ok({"error": str(exc)})
