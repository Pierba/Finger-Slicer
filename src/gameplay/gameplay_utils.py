"""
Gameplay utilities
======================================
This module contains utility functions and classes for gameplay mechanics, including:
- Model loading
- Asset loading
- Projectile management
- Miss mark management
- Game state management
- Rendering functions (miss marks, blade, HUD, game over screen)
- Gameplay helpers (sprite rotation and alpha compositing)
"""
from collections import deque
from config import *
import cv2
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from pathlib import Path
import random
import sounddevice as sd
import threading
from typing import Optional
import urllib.request

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model() -> Path:
    """
    Returns:
        The absolute path to the local `hand_landmarker.task` file.
    """
    if not HAND_MODEL_PATH.exists():
        # Create directory if it doesn't exist
        HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand_landmarker model to {HAND_MODEL_PATH} ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    return HAND_MODEL_PATH

# =============================================================================
# ASSET LOADING
# =============================================================================

def load_assets() -> list[np.ndarray]:
    """
    Loads every RGBA PNG from `ASSETS_DIR` and trims away its transparent padding.

    Sprites are saved by the segmentation step already fitted to `PROJECTILE_MAX_SIZE`
    (longest side), so they load at game scale and need no further downscaling here.

    Returns:
        A list of RGBA sprites as NumPy arrays for the projectiles.
    """
    # List to hold the loaded sprites
    sprites: list[np.ndarray] = []
    for path in ASSETS_DIR.glob("*.png"):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        # Validate the image and verify if it has an alpha channel
        if img is None or img.ndim != 3 or img.shape[2] < 4:
            continue

        # Trim transparent padding for a tight projectile hitbox
        trimmed = trim_rgba(img)
        if trimmed is None:
            continue

        sprites.append(trimmed)

    return sprites


def show_warning(title: str, message: str) -> None:
    """
    Pops up a modal warning dialog and blocks until the user dismisses it.

    The gameplay script is usually launched in its own console window, which
    closes the instant the process exits. A plain `print` therefore vanishes
    before the user can read it, so we surface startup problems through a GUI
    dialog instead. Falls back to the console if no display is available.

    Args:
        title:   The dialog window title.
        message: The body text shown to the user.
    """
    print(f"{title}: {message}")
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()                     # hide the empty root window
        root.attributes("-topmost", True)   # keep the dialog above the console
        messagebox.showwarning(title, message)
        root.destroy()
    except Exception:
        pass

# =============================================================================
# AUDIO
# =============================================================================

# Cache of decoded sounds so each file is read and decoded from disk only once
_sound_cache: dict[Path, tuple[np.ndarray, int]] = {}

# Keep a single output stream open for the whole game in order to play overlapping sounds without glitches or delays
_audio_lock                              = threading.Lock()
_voices: list[list]                      = []      # currently-playing sounds, each a [samples, cursor] pair
_audio_stream: Optional[sd.OutputStream] = None    # the shared sounddevice.OutputStream, opened on first use

def _audio_callback(outdata: np.ndarray, frames: int, time_info, status) -> None:
    """
    Mixes every active voice into the output buffer. Runs on the audio thread.
    
    Args:
        outdata: The output buffer to fill with audio samples.
        frames: The number of frames to be filled in this callback.
        time_info: Timing information provided by sounddevice (not used here).
        status: Callback status provided by sounddevice (not used here).
    """
    outdata.fill(0.0)
    with _audio_lock:
        for voice in _voices:
            samples, cursor = voice
            chunk = samples[cursor:cursor + frames]
            outdata[:len(chunk)] += chunk
            voice[1] = cursor + len(chunk)
        # Drop voices that have finished playing
        _voices[:] = [v for v in _voices if v[1] < len(v[0])]
    # Clamp so several overlapping effects can't sum past full scale and distort
    np.clip(outdata, -1.0, 1.0, out=outdata)


def _decode(path: Path) -> tuple[np.ndarray, int]:
    """
    Decodes an audio file to (samples, samplerate), caching the result.
    
    Args:
        path: Path to the audio file to decode.

    Returns:
        A tuple of (samples, samplerate).
    """
    import soundfile as sf
    if path not in _sound_cache:
        _sound_cache[path] = sf.read(str(path), dtype="float32", always_2d=True)
    return _sound_cache[path]


def _ensure_stream(fs: int, channels: int) -> None:
    """
    Opens the shared output stream once, matching the given audio format.
    
    Args:
        fs: Sample rate of the audio to be played.
        channels: Number of channels of the audio to be played.
    """
    global _audio_stream
    if _audio_stream is None:
        _audio_stream = sd.OutputStream(
            samplerate=fs, channels=channels, callback=_audio_callback,
        )
        _audio_stream.start()


def init_audio() -> None:
    """
    Pre-decodes the game's sound effects and opens the audio device up front.

    Opening the device costs ~200 ms, so warming it here at startup keeps the
    first in-game slice from hitching. Call once before the main loop.
    """
    for path in (BLADE_SLICE_SOUND, GAME_OVER_SOUND):
        samples, fs = _decode(path)
        _ensure_stream(fs, samples.shape[1])


def play_sound(path: Path) -> None:
    """
    Plays a sound effect asynchronously by mixing it into a shared output stream.

    The file is decoded once and cached; each call then just queues the samples
    for the always-open stream to mix, so triggering a sound is cheap and never
    blocks the game loop, even when fired on many consecutive frames.

    Args:
        path: Path to the audio file to play.
    """
    samples, fs = _decode(path)
    _ensure_stream(fs, samples.shape[1])

    # Queue the sound for the callback to mix in on its next block
    with _audio_lock:
        _voices.append([samples, 0])

# =============================================================================
# PROJECTILE
# =============================================================================

def add_outline(sprite: np.ndarray, color: tuple[int, int, int], thickness: int) -> np.ndarray:
    """
    Returns a copy of an RGBA `sprite` with an outline defined by `color` and `thickness`.
    Used to mark special projectiles (red = bomb, yellow = combo).

    Args:
        sprite:    RGBA image as a NumPy array.
        color:     BGR tuple for the outline color.
        thickness: Thickness of the outline in pixels.

    Returns:
        A new RGBA image with the outline applied.
    """
    out = sprite.copy()

    # Extract the alpha channel and find contours to create the outline mask
    _, mask = cv2.threshold(out[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours onto a blank mask to create the outline
    outline = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(outline, contours, -1, 255, thickness, cv2.LINE_AA)

    # Apply the outline color to the output image where the outline mask is present
    where = outline > 0
    out[where, 0] = color[0]
    out[where, 1] = color[1]
    out[where, 2] = color[2]
    out[where, 3] = 255

    return out

class ProjectileKind(Enum):
    """
    The three projectile types:
    - `NORMAL`: sliced once for a point
    - `BOMB`: slicing it is an instant game over
    - `COMBO`: sliced repeatedly, triggers slow-motion, pure bonus
    """
    NORMAL = auto()     # sliced once for a point
    BOMB   = auto()     # slicing it is an instant game over
    COMBO  = auto()     # sliced repeatedly, triggers slow-motion, pure bonus

@dataclass
class Projectile:
    """
    Projectile properties and physics state
    """
    sprite:   np.ndarray        # RGBA image of the projectile
    x:        float             # horizontal position of the projectile center
    y:        float             # vertical position of the projectile center
    vx:       float             # horizontal velocity of the projectile
    vy:       float             # vertical velocity of the projectile
    angle:    float = 0.0       # rotation angle of the projectile
    omega:    float = 0.0       # angular velocity of the projectile
    sliced:   bool  = False     # flags the two halves so they are not re-sliced
    scored:   bool  = False     # flags anything that should NOT count as a miss when it leaves the screen
    kind: ProjectileKind = ProjectileKind.NORMAL  # which of the three projectile types this is
    hits:     int   = 0         # combo-only: how many times it has been hit so far

    def update(self, time_scale: float = 1.0) -> None:
        # time_scale < 1.0 produces the combo slow-motion effect. Scaling
        # both gravity and velocity keeps the trajectory shape identical, just
        # traversed more slowly — what you'd expect from "time slows down"
        self.vy    += GRAVITY     * time_scale
        self.x     += self.vx     * time_scale
        self.y     += self.vy     * time_scale
        self.angle += self.omega  * time_scale

    def draw(self, frame: np.ndarray) -> None:
        # Rotate every frame: cheap at ~180px sprites and avoids tracking a separate cached image
        rotated = rotate_rgba(self.sprite, self.angle)
        blit_rgba(frame, rotated, int(self.x), int(self.y))

    def hitbox(self) -> tuple[int, int, int, int]:
        """
        Axis-aligned rect around the projectile center, shrunk by
        SLICE_HITBOX_SHRINK so the player has to actually cut through the
        visible object rather than swipe near it.
        Uses the unrotated sprite dims — fine for the squarish objects we get
        from segmentation, and much cheaper than a true rotated-poly test.
        """
        h, w = self.sprite.shape[:2]
        bw   = max(1, int(w * SLICE_HITBOX_SHRINK))
        bh   = max(1, int(h * SLICE_HITBOX_SHRINK))
        return (int(self.x - bw // 2), int(self.y - bh // 2), bw, bh)

    def offscreen(self, W: int, H: int) -> bool:
        # `margin` keeps just-barely-spawned projectiles (which start below H) alive
        margin = max(self.sprite.shape[:2])
        return self.y > H + margin or self.x < -margin or self.x > W + margin

def _split(p: Projectile) -> list[Projectile]:
    """
    Splits a projectile into two halves on slice, giving them opposite horizontal kicks and a spin boost.

    Args:
        p: The projectile to be split.

    Returns:
        A list containing the two new Projectile instances representing the halves.
    """
    # The split is vertical down the middle of the sprite, so we slice the RGBA array in half
    w   = p.sprite.shape[1]
    mid = w // 2
    left  = p.sprite[:, :mid].copy()
    right = p.sprite[:, mid:].copy()

    # If either half is fully transparent, we skip the split and return an empty list
    if left.size == 0 or right.size == 0:
        return []

    # The new projectiles inherit the original's position and velocity, but get a horizontal kick and spin boost in opposite directions
    return [
        Projectile(
            sprite=left,
            x=p.x - w / 4, y=p.y,
            vx=p.vx - SLICE_KICK, vy=p.vy - 2.0,
            angle=p.angle, omega=p.omega - SLICE_SPIN_BOOST,
            sliced=True, scored=True,
        ),
        Projectile(
            sprite=right,
            x=p.x + w / 4, y=p.y,
            vx=p.vx + SLICE_KICK, vy=p.vy - 2.0,
            angle=p.angle, omega=p.omega + SLICE_SPIN_BOOST,
            sliced=True, scored=True,
        ),
    ]

# =============================================================================
# MISS MARK
# =============================================================================

@dataclass
class MissMark:
    """
    Short-lived red 'X' drawn where a projectile left the screen unsliced.
    """
    x:   int        # Horizontal position of the mark center
    y:   int        # Vertical position of the mark center
    age: int = 0    # Age in frames; mark is removed when age exceeds MISS_MARK_LIFETIME

# =============================================================================
# GAME STATE
# =============================================================================

@dataclass
class GameState:
    """
    Game state manager holding all variables and methods related to the gameplay loop.
    """
    projectiles:   list[Projectile] = field(default_factory=list)                               # Active projectiles currently on screen
    miss_marks:    list[MissMark]   = field(default_factory=list)                               # Recent miss marks to be rendered and aged out
    trail:         deque            = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))    # Trail of recent fingertip positions for blade rendering and slice detection
    score:         int  = 0     # Player's current score
    misses:        int  = 0     # Count of unsliced projectiles that left the screen; game over when it reaches MAX_MISSES
    frame_count:   int  = 0     # Total frames elapsed since the start of the game; used for timing projectile spawns
    slowmo_frames: int  = 0     # remaining frames for slow-motion effect

    def reset(self):
        """
        Resets the game state to start a new game, clearing all projectiles, miss marks, and trails, and resetting score and misses.
        """
        self.projectiles.clear()
        self.miss_marks.clear()
        self.trail.clear()
        self.score         = 0
        self.misses        = 0
        self.frame_count   = 0
        self.slowmo_frames = 0

    def game_over(self) -> bool:
        """
        Returns:
            True if the game is over (misses reach `MAX_MISSES`), False otherwise.
        """
        return self.misses >= MAX_MISSES

    def update_trail(self, point: Optional[tuple[int, int]]):
        """
        Updates the fingertip trail with the latest detected position

        Args:
            point: The latest detected fingertip position as a tuple (x, y), or None if the hand is not detected.
        """
        # Drop the trail when the hand vanishes
        if point is None:
            self.trail.clear()
        # Add the new point to the trail; the deque will automatically drop the oldest if we exceed TRAIL_LEN
        else:
            self.trail.append(point)

    def maybe_spawn(self, sprites: list[np.ndarray], W: int, H: int):
        """
        Spawns a new projectile at regular intervals defined by `SPAWN_INTERVAL_FRAMES`,
        with random properties and a random sprite took from the provided list.

        Args:
            sprites: List of RGBA sprites to randomly choose from for the new projectile.
            W: Width of the game frame.
            H: Height of the game frame.
        """
        # Only spawn a new projectile every SPAWN_INTERVAL_FRAMES frames
        if self.frame_count % SPAWN_INTERVAL_FRAMES != 0:
            return

        # Picking a random sprite
        sprite = random.choice(sprites)
        # Single roll decides between normal / bomb / combo projectiles
        roll = random.random()
        if roll < BOMB_SPAWN_CHANCE:
            kind = ProjectileKind.BOMB
            sprite = add_outline(sprite, BOMB_OUTLINE_COLOR, BOMB_OUTLINE_THICKNESS)   # Bombs get a red outline
        elif roll < BOMB_SPAWN_CHANCE + COMBO_SPAWN_CHANCE:
            kind = ProjectileKind.COMBO
            sprite = add_outline(sprite, COMBO_OUTLINE_COLOR, COMBO_OUTLINE_THICKNESS) # Combos get a yellow outline
        else:
            kind = ProjectileKind.NORMAL

        # Spawn just below the visible frame so the projectile "rises" into view
        x = random.randint(int(W * 0.15), int(W * 0.85))
        y = H + sprite.shape[0] // 2

        # Arc inward: pick |vx| then throw it toward the center
        speed = random.uniform(*LAUNCH_VX_RANGE)
        vx    = speed if x < W // 2 else -speed
        vy    = random.uniform(*LAUNCH_VY_RANGE)
        omega = random.uniform(*SPIN_RANGE)

        # Create and add the new projectile to the game state
        self.projectiles.append(Projectile(
            sprite=sprite, x=x, y=y, vx=vx, vy=vy, omega=omega,
            kind=kind, scored=(kind is ProjectileKind.COMBO), # Combos are pure bonus so if it leaves the screen unsliced doesn't cost a life
        ))

    def step(self, W: int, H: int):
        """
        Updates the position of all projectiles based on their velocity and gravity, applies slow-motion if active
        and removes any projectiles that have left the screen.
        If a normal projectile leaves the screen unsliced, counts it as a miss and adds a MissMark accordingly.

        Args:
            W: Width of the game frame.
            H: Height of the game frame.
        """
        # Slow-motion is granted by combo hits and decays one real frame per tick
        time_scale = COMBO_SLOWMO_FACTOR if self.slowmo_frames > 0 else 1.0
        if self.slowmo_frames > 0:
            self.slowmo_frames -= 1

        # Update all projectiles and filter out those that have left the screen and handle misses
        survivors: list[Projectile] = []
        for p in self.projectiles:
            p.update(time_scale)
            if p.offscreen(W, H):
                # A whole projectile that left without being sliced costs a life
                # Halves (scored=True) and already-sliced fragments don't
                # Bombs are skipped too
                if not p.scored and p.kind is not ProjectileKind.BOMB:
                    # Increasing the missing counter
                    self.misses += 1

                    # Clamp to the visible frame so the X lands at the screen edge
                    # the projectile escaped through, instead of off-canvas
                    pad = MISS_MARK_SIZE + MISS_MARK_THICKNESS
                    mx = max(pad, min(W - pad, int(p.x)))
                    my = max(pad, min(H - pad, int(p.y)))
                    self.miss_marks.append(MissMark(x=mx, y=my))
                continue

            # Append surviving projectiles from the filter
            survivors.append(p)

        # Update the game state with the surviving projectiles after filtering phase
        self.projectiles = survivors

    def detect_slices(self):
        """
        Fires a slice to a projectile when the most recent fingertip segment:
        - is moving faster than MIN_SLICE_SPEED
        - intersects an un-sliced projectile's hitbox
        """
        # Need at least two points to form a segment and calculate speed
        if len(self.trail) < 2:
            return

        p1, p2 = self.trail[-2], self.trail[-1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]

        # Calculate the speed of the fingertip movement and check if it exceeds the minimum slice speed
        if (dx * dx + dy * dy) ** 0.5 < MIN_SLICE_SPEED:
            return

        # Check each projectile to see if the slice segment intersects its hitbox
        next_projectiles: list[Projectile] = []
        for proj in self.projectiles:
            # Skip already sliced projectiles
            if proj.sliced:
                next_projectiles.append(proj)
                continue

            # Check for intersection between the slice segment and the projectile's hitbox
            inside, _, _ = cv2.clipLine(proj.hitbox(), p1, p2)
            if inside:
                # Play the blade slice sound effect on hit
                play_sound(BLADE_SLICE_SOUND)
                
                match proj.kind:
                    # Slicing a bomb is an instant game over
                    case ProjectileKind.BOMB:
                        # Bump misses to the cap so game_over() flips true on the same frame
                        self.misses = MAX_MISSES
                        next_projectiles.append(proj)

                    # A combo can be hit multiple times, eventually splitting when it reaches COMBO_MAX_HITS
                    case ProjectileKind.COMBO:
                        # Each hit increases the score, refreshes the slow-motion timer, and increments the hit count
                        self.score        += 1
                        self.slowmo_frames = COMBO_SLOWMO_DURATION
                        proj.hits         += 1

                        # On final hit it gets sliced, so the halves get added to the projectile list
                        if proj.hits >= COMBO_MAX_HITS:
                            next_projectiles.extend(_split(proj))

                        # Otherwise add it as a whole projectile for the next frame
                        else:
                            next_projectiles.append(proj)

                    # Normal projectiles are sliced immediately into halves
                    case ProjectileKind.NORMAL:
                        self.score += 1
                        next_projectiles.extend(_split(proj))

            # If the slice does not intersect the projectile, it survives to the next frame as is
            else:
                next_projectiles.append(proj)

        # Update the game state with the new list of projectiles after processing all slices
        self.projectiles = next_projectiles

# =============================================================================
# RENDERING
# =============================================================================

def draw_miss_marks(frame: np.ndarray, miss_marks: list[MissMark]):
    """
    Draws a red X at every recent miss, then age and prune the list.

    Args:
        frame: The current video frame to draw on.
        miss_marks: List of active MissMark instances to be rendered and aged.
    """
    # Draw each miss mark as a red X and increment its age
    for m in miss_marks:
        cv2.line(frame, (m.x - MISS_MARK_SIZE, m.y - MISS_MARK_SIZE), (m.x + MISS_MARK_SIZE, m.y + MISS_MARK_SIZE),
                 MISS_MARK_COLOR, MISS_MARK_THICKNESS, cv2.LINE_AA)
        cv2.line(frame, (m.x - MISS_MARK_SIZE, m.y + MISS_MARK_SIZE), (m.x + MISS_MARK_SIZE, m.y - MISS_MARK_SIZE),
                 MISS_MARK_COLOR, MISS_MARK_THICKNESS, cv2.LINE_AA)
        m.age += 1

    # Remove miss marks that have exceeded their lifetime
    miss_marks[:] = [m for m in miss_marks if m.age < MISS_MARK_LIFETIME]


def draw_blade(frame: np.ndarray, trail: deque):
    """
    Renders the fingertip trail as a tapered white polyline plus a ring at the tip.

    Args:
        frame: The current video frame to draw on.
        trail: Deque of recent fingertip positions.
    """
    # Draw the trail as a series of connected lines, with thickness increasing towards the tip
    pts = list(trail)
    for i in range(1, len(pts)):
        # Older segments are thinner: i makes the line thicker as it increases since pts is ordered oldest -> newest
        cv2.line(frame, pts[i - 1], pts[i], (255, 255, 255), max(1, i), cv2.LINE_AA)

    # Draw a green circle at the tip of the blade
    if pts:
        cv2.circle(frame, pts[-1], 12, (0, 255, 0), 2)


def draw_hud(frame: np.ndarray, state: GameState, W: int):
    """
    Draws the score and misses count on the top-right corner of the screen.

    Args:
        frame: The current video frame to draw on.
        state: The current game state containing score and misses information.
        W: The width of the game frame, used to position the HUD elements.
    """
    cv2.putText(frame, f"Score: {state.score}", (W - 230, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Misses: {state.misses}/{MAX_MISSES}", (W - 230, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


def draw_game_over(frame: np.ndarray, state: GameState, W: int, H: int):
    """
    Draws the game over screen with the final score and instructions to restarting or quitting the game.

    Args:
        frame: The current video frame to draw on.
        state: The current game state containing score and misses information.
        W: The width of the game frame, used to position the HUD elements.
        H: The height of the game frame, used to position the HUD elements.
    """
    title = "GAME OVER"
    sub   = f"Final score: {state.score}   |   R = restart   Q = quit"

    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
    (sw, _), _ = cv2.getTextSize(sub,   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

    # Black stroke + white fill so the message stays legible over any webcam background
    cv2.putText(frame, title, ((W - tw) // 2, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 6, cv2.LINE_AA)

    cv2.putText(frame, title, ((W - tw) // 2, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.putText(frame, sub, ((W - sw) // 2, H // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)

    cv2.putText(frame, sub, ((W - sw) // 2, H // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

# =============================================================================
# GAMEPLAY HELPERS
# =============================================================================

def trim_rgba(sprite: np.ndarray) -> Optional[np.ndarray]:
    """
    Crops an RGBA `sprite` tightly to the bounding box of its non-zero alpha pixels.
    Strips the padding so projectiles get a tight hitbox.

    Returns:
        The cropped RGBA sprite as a NumPy array, or None if the alpha channel is fully empty.
    """
    # If the image doesn't have an alpha channel, return it as is
    if sprite.ndim != 3 or sprite.shape[2] < 4:
        return sprite

    # Find the bounding box of non-transparent pixels in the alpha channel
    ys, xs = np.where(sprite[:, :, 3] > 0)
    if len(xs) == 0:
        return None

    # Crop the sprite to the bounding box and return it
    return sprite[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


def rotate_rgba(sprite: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotates `sprite` (BGRA) around its center by `angle_deg` degrees and expands
    the output canvas so the rotated image fits without any corner clipping.

    Args:
        sprite: RGBA image as a NumPy array.
        angle_deg: Rotation angle in degrees (counterclockwise).

    Returns:
        The rotated RGBA image as a NumPy array, with an expanded canvas to avoid clipping.
    """
    # Get the rotation matrix for the specified angle around the center of the sprite
    h, w = sprite.shape[:2]
    M    = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)

    # Expand canvas when the rotated bounding box is larger than the original
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w    = int(h * sin + w * cos)
    new_h    = int(h * cos + w * sin)

    # Shift so the rotated image is centered inside the expanded canvas
    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2

    # Rotate the image using the computed matrix and expanded canvas size, filling empty areas with transparent pixels
    return cv2.warpAffine(
        sprite, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def blit_rgba(frame_bgr: np.ndarray, sprite_bgra: np.ndarray, cx: int, cy: int):
    """
    Alpha-composite `sprite_bgra` onto `frame_bgr` in place, centered at (`cx`, `cy`).
    Handles clipping at all four frame edges.

    Args:
        frame_bgr: The destination BGR image (game frame) as a NumPy array.
        sprite_bgra: The source RGBA image (projectile sprite) as a NumPy array.
        cx: The x-coordinate of the center position where the sprite should be blitted on the frame.
        cy: The y-coordinate of the center position where the sprite should be blitted on the frame.
    """
    # Get dimensions of the sprite and frame for clipping calculations
    sh, sw = sprite_bgra.shape[:2]
    fh, fw = frame_bgr.shape[:2]

    # Target rect in frame coordinates
    x0, y0 = cx - sw // 2, cy - sh // 2
    x1, y1 = x0 + sw, y0 + sh

    # Clip target to the frame, mirroring the clip back onto the source rect
    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    src_x1 = sw - max(0, x1 - fw)
    src_y1 = sh - max(0, y1 - fh)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return

    # Corresponding source rect in sprite coordinates, after clipping
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    # Extract the relevant sprite region and perform alpha compositing onto the frame
    sprite_clip = sprite_bgra[src_y0:src_y1, src_x0:src_x1]
    alpha       = sprite_clip[:, :, 3:4].astype(np.float32) / 255.0
    bgr_src     = sprite_clip[:, :, :3].astype(np.float32)

    # Alpha composite the sprite onto the frame in place
    roi = frame_bgr[dst_y0:dst_y1, dst_x0:dst_x1]
    np.copyto(roi, (bgr_src * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8))