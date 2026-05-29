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
from   typing import Optional

import time

import cv2

import mediapipe              as mp
from   mediapipe.tasks        import python as mp_python
from   mediapipe.tasks.python import vision as mp_vision

from config         import *
from gameplay_utils  import *

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
