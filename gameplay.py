"""
Finger Slicer — Fruit-Ninja-style gameplay loop.
=================================================
Pulls RGBA sprites from `assets/` (produced by segment_objects.py), launches
them across the screen with simple projectile physics, and lets the player
slice them by moving their index fingertip (tracked via MediaPipe).

Pipeline per frame
------------------
  webcam read -> mirror -> MediaPipe detect -> update trail
              -> maybe spawn -> step physics -> detect slices
              -> draw sprites + blade + HUD -> imshow

Controls
--------
  Move your hand quickly through a projectile to slice it.
  R         Restart (after game over)
  Q / Esc   Quit
"""
from   collections import deque
from   dataclasses import dataclass, field
from   typing      import Optional

import random
import time

import cv2
import numpy  as np

import mediapipe              as mp
from   mediapipe.tasks        import python as mp_python
from   mediapipe.tasks.python import vision as mp_vision

from config         import *
from segment_utils  import blit_rgba, rotate_rgba, trim_rgba
from gameplay_utils import load_model

# =============================================================================
# ASSET LOADING
# =============================================================================

def load_assets() -> list[np.ndarray]:
    """
    Load every RGBA PNG from ASSETS_DIR, trim away transparent padding, and
    downscale so the longest side <= PROJECTILE_MAX_SIZE.
    """
    sprites: list[np.ndarray] = []
    for path in ASSETS_DIR.glob("*.png"):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] < 4:
            continue
        trimmed = trim_rgba(img)
        if trimmed is None:
            continue
        h, w  = trimmed.shape[:2]
        scale = PROJECTILE_MAX_SIZE / max(h, w)
        if scale < 1.0:
            trimmed = cv2.resize(trimmed, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        sprites.append(trimmed)
    return sprites


# =============================================================================
# PROJECTILE
# =============================================================================

def add_outline(sprite: np.ndarray, color: tuple[int, int, int], thickness: int) -> np.ndarray:
    """
    Return a copy of an RGBA sprite with a coloured contour traced around its
    silhouette — used to mark special projectiles (red = bomb, yellow = combo).
    Outline pixels are forced fully opaque so they stay visible even where the
    original sprite was transparent.
    """
    out = sprite.copy()
    _, mask = cv2.threshold(out[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    outline = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(outline, contours, -1, 255, thickness, cv2.LINE_AA)

    where = outline > 0
    out[where, 0] = color[0]
    out[where, 1] = color[1]
    out[where, 2] = color[2]
    out[where, 3] = 255
    return out


@dataclass
class Projectile:
    """
    A single sprite flying through the scene. Position and velocity are in
    pixel space; angle/omega are in degrees and degrees-per-frame.
    `sliced` flags the two halves so they are not re-sliced; `scored` flags
    anything that should NOT count as a miss when it leaves the screen
    (everything sliced + every half spawned from a slice).
    `is_bomb` projectiles end the game instantly if the player slices them.
    `is_combo` projectiles can be sliced repeatedly and trigger slow-motion.
    """
    sprite:   np.ndarray
    x:        float
    y:        float
    vx:       float
    vy:       float
    angle:    float = 0.0
    omega:    float = 0.0
    sliced:   bool  = False
    scored:   bool  = False
    is_bomb:  bool  = False
    is_combo: bool  = False
    hits:     int   = 0     # combo-only: how many times it has been hit so far

    def update(self, time_scale: float = 1.0) -> None:
        # time_scale < 1.0 produces the combo slow-motion effect. Scaling
        # both gravity and velocity keeps the trajectory shape identical, just
        # traversed more slowly — what you'd expect from "time slows down".
        self.vy    += GRAVITY     * time_scale
        self.x     += self.vx     * time_scale
        self.y     += self.vy     * time_scale
        self.angle += self.omega  * time_scale

    def draw(self, frame: np.ndarray) -> None:
        # Rotate every frame: cheap at ~180px sprites and avoids tracking a separate cached image.
        rotated = rotate_rgba(self.sprite, self.angle)
        blit_rgba(frame, rotated, int(self.x), int(self.y))

    def hitbox(self) -> tuple[int, int, int, int]:
        """
        Axis-aligned rect around the projectile centre, shrunk by
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
        # `margin` keeps just-barely-spawned projectiles (which start below H) alive.
        margin = max(self.sprite.shape[:2])
        return self.y > H + margin or self.x < -margin or self.x > W + margin


def _split(p: Projectile) -> list[Projectile]:
    """
    v1 slice effect: cut the sprite vertically down its centre and return two
    half-projectiles that fly apart horizontally with extra spin. The cut is
    along the sprite's local axis (not the blade's), so it's not a true
    blade-aligned slice — but the visual reads as "the object came apart" and
    it's a lot less code than computing the rotated cut line.
    """
    w   = p.sprite.shape[1]
    mid = w // 2
    left  = p.sprite[:, :mid].copy()
    right = p.sprite[:, mid:].copy()
    if left.size == 0 or right.size == 0:
        return []

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
    """Short-lived red 'X' drawn where a projectile left the screen unsliced."""
    x:   int
    y:   int
    age: int = 0


# =============================================================================
# GAME STATE
# =============================================================================

@dataclass
class GameState:
    projectiles:   list[Projectile] = field(default_factory=list)
    miss_marks:    list[MissMark]   = field(default_factory=list)
    trail:         deque            = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))
    score:         int  = 0
    misses:        int  = 0
    frame_count:   int  = 0
    slowmo_frames: int  = 0       # remaining frames of combo-induced slow-motion

    def reset(self) -> None:
        self.projectiles.clear()
        self.miss_marks.clear()
        self.trail.clear()
        self.score         = 0
        self.misses        = 0
        self.frame_count   = 0
        self.slowmo_frames = 0

    def game_over(self) -> bool:
        return self.misses >= MAX_MISSES

    def update_trail(self, point: Optional[tuple[int, int]]) -> None:
        # Drop the trail when the hand vanishes: otherwise the next reappearance
        # would join a stale point to a fresh one and slice everything between them.
        if point is None:
            self.trail.clear()
        else:
            self.trail.append(point)

    def maybe_spawn(self, sprites: list[np.ndarray], W: int, H: int) -> None:
        if self.frame_count % SPAWN_INTERVAL_FRAMES != 0:
            return
        sprite = random.choice(sprites)
        # Single roll decides between normal / bomb / combo so the probabilities
        # are exclusive and easy to reason about.
        roll     = random.random()
        is_bomb  = roll < BOMB_SPAWN_CHANCE
        is_combo = not is_bomb and roll < BOMB_SPAWN_CHANCE + COMBO_SPAWN_CHANCE
        if is_bomb:
            sprite = add_outline(sprite, BOMB_OUTLINE_COLOR, BOMB_OUTLINE_THICKNESS)
        elif is_combo:
            sprite = add_outline(sprite, COMBO_OUTLINE_COLOR, COMBO_OUTLINE_THICKNESS)
        # Spawn just below the visible frame so the projectile "rises" into view.
        x = random.randint(int(W * 0.15), int(W * 0.85))
        y = H + sprite.shape[0] // 2

        # Arc inward: pick |vx| then sign it toward the centre.
        speed = random.uniform(*LAUNCH_VX_RANGE)
        vx    = speed if x < W // 2 else -speed
        vy    = random.uniform(*LAUNCH_VY_RANGE)
        omega = random.uniform(*SPIN_RANGE)
        # Combos are pure bonus: a combo that leaves unsliced should not cost a
        # life, so we mark it scored at spawn time.
        self.projectiles.append(Projectile(
            sprite=sprite, x=x, y=y, vx=vx, vy=vy, omega=omega,
            is_bomb=is_bomb, is_combo=is_combo, scored=is_combo,
        ))

    def step(self, W: int, H: int) -> None:
        # Slow-motion is granted by combo hits and decays one real frame per tick.
        time_scale = COMBO_SLOWMO_FACTOR if self.slowmo_frames > 0 else 1.0
        if self.slowmo_frames > 0:
            self.slowmo_frames -= 1

        survivors: list[Projectile] = []
        for p in self.projectiles:
            p.update(time_scale)
            if p.offscreen(W, H):
                # A whole projectile that left without being sliced costs a life.
                # Halves (scored=True) and already-sliced fragments don't.
                # Bombs are skipped too: dodging one is the *correct* play, so
                # we don't punish/mark it.
                if not p.scored and not p.is_bomb:
                    self.misses += 1
                    # Clamp to the visible frame so the X lands at the screen edge
                    # the projectile escaped through, instead of off-canvas.
                    pad = MISS_MARK_SIZE + MISS_MARK_THICKNESS
                    mx = max(pad, min(W - pad, int(p.x)))
                    my = max(pad, min(H - pad, int(p.y)))
                    self.miss_marks.append(MissMark(x=mx, y=my))
                continue
            survivors.append(p)
        self.projectiles = survivors

    def detect_slices(self) -> None:
        """
        A slice fires when the most recent fingertip segment (a) is moving
        faster than MIN_SLICE_SPEED and (b) intersects an un-sliced
        projectile's hitbox. cv2.clipLine does the segment-rect test for us.
        """
        if len(self.trail) < 2:
            return
        p1, p2 = self.trail[-2], self.trail[-1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if (dx * dx + dy * dy) ** 0.5 < MIN_SLICE_SPEED:
            return

        next_projectiles: list[Projectile] = []
        for proj in self.projectiles:
            if proj.sliced:
                next_projectiles.append(proj)
                continue
            inside, _, _ = cv2.clipLine(proj.hitbox(), p1, p2)
            if inside:
                if proj.is_bomb:
                    # Slicing a bomb is an instant game over: bump misses to
                    # the cap so game_over() flips true on the same frame.
                    self.misses = MAX_MISSES
                    next_projectiles.append(proj)
                elif proj.is_combo:
                    # Combo: score, refresh slow-motion, and keep it alive so
                    # the player can keep hitting it. MIN_SLICE_SPEED in the
                    # outer check already prevents a stationary finger from
                    # racking up free points.
                    self.score        += 1
                    self.slowmo_frames = COMBO_SLOWMO_DURATION
                    proj.hits         += 1
                    # Final hit: split it like a regular fruit so the player
                    # gets the satisfying "it finally came apart" feedback.
                    if proj.hits >= COMBO_MAX_HITS:
                        next_projectiles.extend(_split(proj))
                    else:
                        next_projectiles.append(proj)
                else:
                    self.score += 1
                    next_projectiles.extend(_split(proj))
            else:
                next_projectiles.append(proj)
        self.projectiles = next_projectiles


# =============================================================================
# RENDERING
# =============================================================================

def draw_miss_marks(frame: np.ndarray, miss_marks: list[MissMark]) -> None:
    """Draw a red X at every recent miss, then age and prune the list."""
    s = MISS_MARK_SIZE
    for m in miss_marks:
        cv2.line(frame, (m.x - s, m.y - s), (m.x + s, m.y + s),
                 MISS_MARK_COLOR, MISS_MARK_THICKNESS, cv2.LINE_AA)
        cv2.line(frame, (m.x - s, m.y + s), (m.x + s, m.y - s),
                 MISS_MARK_COLOR, MISS_MARK_THICKNESS, cv2.LINE_AA)
        m.age += 1
    miss_marks[:] = [m for m in miss_marks if m.age < MISS_MARK_LIFETIME]


def draw_blade(frame: np.ndarray, trail: deque) -> None:
    """Render the fingertip trail as a tapered white polyline plus a ring at the tip."""
    pts = list(trail)
    for i in range(1, len(pts)):
        # Older segments are thinner: i grows with recency since pts is ordered oldest -> newest.
        cv2.line(frame, pts[i - 1], pts[i], (255, 255, 255), max(1, i), cv2.LINE_AA)
    if pts:
        cv2.circle(frame, pts[-1], 12, (0, 255, 0), 2)


def draw_hud(frame: np.ndarray, state: GameState, W: int) -> None:
    cv2.putText(frame, f"Score: {state.score}", (W - 230, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Misses: {state.misses}/{MAX_MISSES}", (W - 230, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)


def draw_game_over(frame: np.ndarray, state: GameState, W: int, H: int) -> None:
    title = "GAME OVER"
    sub   = f"Final score: {state.score}   |   R = restart   Q = quit"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
    (sw, _), _ = cv2.getTextSize(sub,   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    # Black stroke + white fill so the message stays legible over any webcam background.
    cv2.putText(frame, title, ((W - tw) // 2, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, title, ((W - tw) // 2, H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, sub, ((W - sw) // 2, H // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, sub, ((W - sw) // 2, H // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    sprites = load_assets()
    if not sprites:
        print(f"No RGBA assets found in '{ASSETS_DIR}'. Run segment_objects.py first.")
        return
    print(f"Loaded {len(sprites)} asset(s) from '{ASSETS_DIR}'")

    # MediaPipe hand-landmarker setup (mirrors finger_tracker.py).
    model_path = load_model()
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=HAND_DETECT_CONFIDENCE,
        min_tracking_confidence=HAND_TRACK_CONFIDENCE,
    )

    # Index 0 = default system webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    # Request the desired capture mode
    # OpenCV silently picks the closest one the camera actually supports
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam mode: {H}x{W} @ {FPS:.1f} fps "
          f"(requested {CAM_WIDTH}x{CAM_HEIGHT} @ {CAM_FPS} fps)")
    
    state = GameState()
    start = time.monotonic()
    WIN   = "Finger Slicer (Q to quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Mirror so the on-screen image acts like a mirror to the player.
            frame = cv2.flip(frame, 1)

            # == fingertip detection ===================================
            rgba         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result       = landmarker.detect_for_video(mp_image, timestamp_ms)

            tip: Optional[tuple[int, int]] = None
            if result.hand_landmarks:
                lm  = result.hand_landmarks[0][INDEX_FINGERTIP]
                tip = (int(lm.x * W), int(lm.y * H))

            # == game step =============================================
            if not state.game_over():
                state.update_trail(tip)
                state.maybe_spawn(sprites, W, H)
                state.step(W, H)
                state.detect_slices()
                state.frame_count += 1

            # == render ================================================
            for p in state.projectiles:
                p.draw(frame)
            draw_miss_marks(frame, state.miss_marks)
            draw_blade(frame, state.trail)
            draw_hud(frame, state, W)
            if state.game_over():
                draw_game_over(frame, state, W, H)

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if state.game_over() and key == ord('r'):
                state.reset()

            # Allow closing the window via the [x] button.
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()