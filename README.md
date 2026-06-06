# 🔪 Finger Slicer

> A Fruit-Ninja-style game you play with your bare hands — and you bring your own fruit.

Finger Slicer is an interactive webcam game powered by computer vision. Instead of shipping a
fixed set of fruits, it lets you **turn any photo into playable sprites**: point a segmentation
model at an image, it cuts out the objects it finds, and those objects become the things you
slice on screen by waving your index finger in front of your webcam.

There are no controllers and no mouse during play — your **index fingertip is the blade**,
tracked in real time through your camera.

---

## 🎯 Goal of the project

Finger Slicer combines two computer-vision pipelines into one small game:

1. **Object segmentation** — extract clean, background-free objects from an everyday photo and
   save them as transparent PNG sprites.
2. **Hand tracking** — follow your index fingertip through the webcam and use its motion to
   slice those sprites as they fly across the screen.

The result is a game whose content is whatever you feed it: photograph the fruit on your kitchen
table, the tools on your desk, or your collection of sneakers, and slice them in mid-air.

---

## 🧩 How it works

The project is built around two stages that you run in order.

### Stage 1 — Segment objects into sprites

You give the tool a source image and pick a model. It detects the objects, removes their
backgrounds pixel-by-pixel, and saves each one as a `180×180` (game-scale) transparent PNG into `assets/`.
A composite **preview** image (showing every detected object with a coloured mask and label) is
written to `previews/` so you can see what was found.

Three segmentation modes are available:

| Mode | Model | What it does |
| --- | --- | --- |
| **YOLO (auto)** | YOLO26-seg | Automatic instance segmentation — detects and cuts out known object classes in one pass. *(default)* |
| **SAM (auto)** | SAM2 | Fully automatic — generates every mask it can, then filters out noise, background, and overlapping sub-parts. |
| **SAM (interactive)** | SAM2 | Click-to-segment — left-click the parts of an object to keep, right-click to exclude, and save the result. |

> ⚡ **Hardware note:** **YOLO** is lightweight and runs comfortably on **CPU**, so it's the
> better pick on a modest laptop. **SAM2** is far heavier — it really wants a **CUDA GPU or
> Apple Silicon (MPS)** device to be usable; on plain CPU it works but can be very slow.

In the auto modes you **review each detected object** before it's kept: a preview pops up over a
checkerboard background and you press `Y` to save it or any other key to skip.

Under the hood, every mask goes through the same clean-up pipeline (`upscale → binarise →
morphological refinement → keep largest connected component → tight alpha crop`) so the sprites
have crisp edges and no stray pixels.

### Stage 2 — Play the game

The game loads every sprite from `assets/`, then launches them across the screen as spinning
projectiles with simple gravity-based physics. Your webcam feed is mirrored and shown live, and
**MediaPipe** tracks your hand to find your index fingertip every frame.

Move your fingertip quickly (but not too much or it is difficult to track) through a projectile and you slice it in two. The recent positions of
your fingertip form a glowing "blade" trail, and a slice only registers when that trail is moving
faster than a minimum speed — so resting your finger on screen won't accidentally cut anything.

**Per-frame loop:**

```
webcam read → mirror → MediaPipe hand detect → update fingertip trail
            → maybe spawn projectile → step physics → detect slices
            → draw sprites + blade + HUD → display
```

#### Projectile types

| Type | Marking | Behaviour |
| --- | --- | --- |
| **Normal** | none | Slice it for **+1** point. It splits cleanly into two halves. |
| 💣 **Bomb** | red outline | **Do not slice it!** Cutting a bomb is an instant game over. |
| ⭐ **Combo** | yellow outline | Hit it repeatedly for points and trigger a **slow-motion** effect. It finally splits after enough hits, and never costs you a life if you miss it. |

You have **3 lives**. Letting a normal projectile fall off-screen unsliced marks a red ✗ and
costs a life — three misses ends the run.

> 🔊 **Sound:** a slice sound plays every time you cut a projectile, and a game-over sound
> plays once when your last life is lost. Effects are pre-decoded and mixed through a single
> always-open audio stream, so overlapping slices never glitch or lag.

---

## 📦 Project structure

```
Finger-Slicer/
├── launcher.py                       # GUI launcher (entry point)
├── config.py                         # All tunable constants (physics, thresholds, models…)
├── requirements.txt
├── src/
│   ├── segmentation/
│   │   ├── segment_objects.py        # Stage 1: YOLO / SAM segmentation modes
│   │   └── segment_utils.py          # Mask processing, CLI, saving helpers
│   └── gameplay/
│       ├── gameplay.py               # Stage 2: main game loop
│       └── gameplay_utils.py         # Projectiles, physics, rendering, audio, hand-tracking setup
├── sounds/                           # Gameplay sound effects (slice, game over)
├── assets/                           # Generated sprites land here (the game's "fruit")
├── previews/                         # Segmentation preview maps
└── .temp/                            # Auto-downloaded model weights
```

Model weights are **downloaded automatically on first use** and cached in `.temp/`
(the MediaPipe hand landmarker, plus the YOLO/SAM weights pulled by Ultralytics).

---

## 🚀 Getting started

### Requirements

- **Python 3.10+** (the code uses `match` statements and modern type syntax)
- A **webcam**
- The dependencies in `requirements.txt` (PyTorch, Ultralytics, MediaPipe, OpenCV, CustomTkinter,
  sounddevice, SoundFile, NumPy)

A CUDA GPU or Apple Silicon (MPS) will speed up segmentation, but everything runs on CPU too —
the device is detected automatically.

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Finger-Slicer

# (Recommended) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run it

The easiest way is the **launcher**, a small GUI that wires both stages together:

```bash
python launcher.py
```

From there you can:

- **Segment Objects** — pick an image, choose a model/mode, and generate sprites. When
  segmentation finishes, the **Play Game** button lights up automatically.
- **Play Game** — jump straight into the game using whatever sprites are already in `assets/`.

> The game needs at least one valid sprite in `assets/` to play. If the folder is empty it will
> tell you to run *Segment Objects* first.

---

## 🎮 Controls

### In-game

| Action | Control |
| --- | --- |
| Slice | Move your **index fingertip** quickly through a projectile |
| Restart (after game over) | `R` |
| Quit | `Q` or `Esc` |

### Interactive segmentation (SAM)

| Action | Control |
| --- | --- |
| Mark a point to **include** | Left-click |
| Mark a point to **exclude** | Right-click |
| Run segmentation | `S` |
| Save the mask as a PNG | `Enter` |
| Clear points & retry | `C` |
| Undo last point | `Z` |
| Quit | `Q` or `Esc` |

---

## 🛠️ Command-line usage

The segmentation stage can be run directly without the launcher:

```bash
# YOLO auto (default)
python src/segmentation/segment_objects.py photo.jpg

# SAM fully-automatic
python src/segmentation/segment_objects.py photo.jpg --model-type sam

# SAM interactive (click to segment)
python src/segmentation/segment_objects.py photo.jpg --model-type sam -i

# Tune detection / pick a different model
python src/segmentation/segment_objects.py photo.jpg --conf 0.1
python src/segmentation/segment_objects.py photo.jpg --yolo-model yolo26l-seg.pt
```

| Flag | Description |
| --- | --- |
| `image` | Input image (JPG, PNG, …) — **required** |
| `--model-type {yolo,sam}` | Segmentation backend (default: `yolo`) |
| `-i`, `--interactive` | Click-to-segment mode (SAM only) |
| `--conf` | Detection confidence for auto modes (default: `0.05`; raise to `0.25+` for scene photos) |
| `--yolo-model` / `--sam-model` | Override the model weights |

The game itself takes no arguments:

```bash
python src/gameplay/gameplay.py
```

> 💡 **Tip:** For best results, segment flat-lay or top-down product photos with a clean
> background. They produce tidy, well-separated sprites — and low confidence values
> (`0.05`–`0.15`) help YOLO pick up objects it would otherwise miss.

---

## ⚙️ Configuration

Almost every tunable lives in [`config.py`](config.py), grouped by area — no need to touch the
logic. A few you might want to play with:

| Constant | Effect |
| --- | --- |
| `SPAWN_INTERVAL_FRAMES` | How often projectiles spawn — lower = harder |
| `MAX_MISSES` | Lives before game over (default `3`) |
| `BOMB_SPAWN_CHANCE` / `COMBO_SPAWN_CHANCE` | How often bombs / combos appear |
| `MIN_SLICE_SPEED` | How fast your finger must move to cut |
| `PROJECTILE_MAX_SIZE` | On-screen size of the flying sprites |
| `GRAVITY`, `LAUNCH_VX_RANGE`, `LAUNCH_VY_RANGE` | Projectile physics and arc |
| `CAM_WIDTH`, `CAM_HEIGHT`, `CAM_FPS` | Requested webcam capture mode |
| `SMALL_OBJECT_THRESHOLD`, `BACKGROUND_THRESHOLD`, `SUBPART_OVERLAP_THRESHOLD` | SAM auto-mode filtering |
| `BLADE_SLICE_SOUND`, `GAME_OVER_SOUND` | Sound-effect files used during play |

---

## 🧠 Tech stack

- **[Ultralytics](https://github.com/ultralytics/ultralytics)** — YOLO26-seg & SAM2 segmentation models
- **[MediaPipe](https://github.com/google-ai-edge/mediapipe)** — real-time hand-landmark tracking
- **[OpenCV](https://opencv.org/)** — webcam capture, image processing, and rendering
- **[PyTorch](https://pytorch.org/)** — model inference backend (CUDA / MPS / CPU)
- **[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)** — the launcher GUI
- **[sounddevice](https://python-sounddevice.readthedocs.io/)** + **[SoundFile](https://python-soundfile.readthedocs.io/)** — low-latency sound-effect playback
- **[NumPy](https://numpy.org/)** — mask math and alpha compositing

---

*Snap a photo, slice it to pieces. 🍉*
