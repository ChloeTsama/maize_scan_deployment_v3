# MaizeScan — Deployment Package (v3)

Flask web application that diagnoses maize (corn) leaf diseases from a
photo, using a fine-tuned VGG16 CNN, and **rejects any image that is not
a maize leaf** — including photos of people, animals, and everyday
objects — before a diagnosis is ever produced.

Author: Brice Gaetan Nono Youmbi | Roll No. 202211043
Supervisor: Jonas Niyitegeka | Kigali Independent University ULK, Data
Science, 2025/2026

---

## What's new in v3: the 3-stage "maize leaf, and only a maize leaf" gate

Every uploaded image must pass **three independent checks**, in order,
before it reaches a disease prediction. Failing any one of them stops
the pipeline and shows the user a clear reason — it never falls through
to a guessed diagnosis.

| Stage | What it does | Rejects |
|---|---|---|
| **Stage 0 — General content gate** (`utils/general_image_gate.py`) | Runs a pretrained MobileNetV2 (ImageNet, 1000 classes) on the image and checks the top prediction against curated animal / person-indicator / man-made-object keyword lists. | Photos of people, animals, vehicles, furniture, electronics, buildings, and everyday objects |
| **Stage 1 — Visual feature check** (`utils/leaf_validator.py::validate_visual`) | Fast (<5ms) colour/texture heuristics: green dominance, saturation, texture variance, blank-image detection, colour-distribution extremes. | Blank/solid-colour images, non-green scenes (sky, soil, paper), washed-out/greyscale images |
| **Stage 2 — CNN entropy gate** (`utils/leaf_validator.py::validate_cnn`) | After the disease CNN runs, checks its own confidence and the Shannon entropy of its softmax output. | Anything the disease model itself has no confident, clear pattern for |

All three stages **fail open on internal error** — e.g. if Stage 0's
model cannot be downloaded, it logs a warning and passes the image
through to Stage 1/2 rather than crashing the app. This means the app
never goes down because of the general-content gate, but real leaf
photos are still protected by Stage 1 and Stage 2 either way.

### Why a general-purpose classifier (Stage 0) was added

The original 2-stage validator relied only on colour and texture rules.
That reliably rejects blank/wrong-colour images, but a **green,
textured, non-leaf object** — a person in a green shirt against grass, a
parrot, a green plastic toy — could still pass pure colour/texture
checks. Stage 0 adds a pretrained, general-purpose image classifier
(MobileNetV2 on ImageNet) as an extra, independent gate specifically to
catch people, animals, and man-made objects, which is exactly the
requirement this package was built to satisfy.

MobileNetV2 was chosen for this gate (not VGG16) because it is small
(~14 MB), fast on CPU, and this task only needs a coarse "is this
obviously not a plant?" signal — not a fine-grained classifier.

---

## Project structure

```
maizescan/
├── app.py                        Flask application & routes
├── requirements.txt
├── runtime.txt                   Python version pin (Render)
├── Procfile                      gunicorn start command
├── render.yaml                   Render deployment config
├── DIAGNOSTIC_COLAB_CELL.py      Run in Colab to export class_indices.json
├── model/
│   ├── README.md                 How to place your trained model here
│   ├── class_indices.json        Maps model output index -> class name
│   └── vgg16_maize_best.h5       ⚠ NOT included — add your trained model
├── utils/
│   ├── leaf_image.py             Image loading & preprocessing
│   ├── cnn_model.py               Loads/runs the disease CNN
│   ├── leaf_validator.py         MaizeLeafValidator — Stages 1 & 2, orchestrates Stage 0
│   ├── general_image_gate.py     Stage 0 — MobileNetV2 person/animal/object gate (NEW)
│   ├── diagnosis.py              Builds the final diagnosis response
│   └── disease_data.py           Disease info + class-order loading
├── templates/                    Jinja2 HTML templates
├── static/                       CSS, JS, uploaded images
└── tests/
    └── test_validator_pipeline.py   Offline unit tests for Stages 0-2 (NEW)
```

---

## Setup

1. **Add your trained model.** Place your trained Keras model at
   `model/vgg16_maize_best.h5` and the matching `model/class_indices.json`
   (see `model/README.md` for exactly how to generate this from Colab).
   Without these, the app runs in **DEMO MODE** (dummy predictions) so
   the UI can still be reviewed and the validator gate still runs.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run locally:**
   ```bash
   python app.py
   # then open http://localhost:5000
   ```
   The first prediction request (or app startup, if `WARM_GENERAL_GATE`
   is left at its default `true`) downloads MobileNetV2's ImageNet
   weights (~14 MB) to the Keras cache directory. This requires internet
   access once; after that it's cached locally.

4. **Environment variables (optional):**
   | Variable | Default | Purpose |
   |---|---|---|
   | `SECRET_KEY` | `maizescan-ulk-2026` | Flask secret key |
   | `WARM_GENERAL_GATE` | `true` | Preload Stage 0's MobileNetV2 at startup instead of on first request |
   | `PORT` | `5000` | Server port |

5. **Deploy to Render:** push this project to a Git repo and connect it
   in Render, or use `render.yaml` directly. `Procfile` and
   `requirements.txt` are already set up for a Python web service.

---

## Verifying the reject-non-maize behaviour

`tests/test_validator_pipeline.py` contains offline unit tests (no
trained model or internet required for Stage 1/2) that generate
synthetic images — solid colour, random noise, blue "sky", grey
"pavement" — and assert Stage 1 rejects them. Run:

```bash
python -m pytest tests/test_validator_pipeline.py -v
```

To manually verify Stage 0 against real photos (a person, a dog, a car,
etc.), start the app and upload them through the `/` upload form, or
POST to `/api/predict` and inspect the JSON `reason_code`
(`detected_animal`, `detected_person`, `detected_object`,
`not_green`, `blank_image`, `low_texture`, `wrong_color_distribution`,
`low_saturation`, `cnn_uncertain`) in the response.

```bash
curl -F "leaf_image=@/path/to/dog_photo.jpg" http://localhost:5000/api/predict
```

---

## Known limitation of Stage 0

Stage 0's keyword lists are curated from ImageNet's 1000 classes and are
not exhaustive — a very unusual object with no close ImageNet match
could still pass through to Stage 1/2. Stage 1 (colour/texture) and
Stage 2 (CNN entropy) remain active as a second and third line of
defence in exactly that situation, which is why the three stages are
layered rather than relying on any single check alone.
# maize_scan_deployment_v3
