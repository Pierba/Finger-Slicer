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
  M         Mute / unmute audio
  R         Restart (after game over)
  Q / Esc   Quit
"""
from pathlib import Path
import sys

# Adjust the import path to include the project root
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from config import *
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from src.gameplay.gameplay_utils import *
import time
from typing import Optional

# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main game loop.
    """
    # Load projectile sprites from the assets directory
    sprites = load_assets()
    if not sprites:
        # Without sprites there is nothing to throw, so warn and exit out
        png_count = len(list(ASSETS_DIR.glob("*.png"))) if ASSETS_DIR.exists() else 0
        if png_count == 0:
            message = (f"No images found in:\n{ASSETS_DIR}\n\n"
                       "Run 'Segment Objects' first to create some sprites.")
        else:
            message = (f"Found {png_count} image(s) in:\n{ASSETS_DIR}\n"
                       "but none are valid RGBA sprites.\n\n"
                       "Re-run 'Segment Objects' to regenerate them.")
        show_warning("Finger Slicer - no playable images", message)
        return

    print(f"Loaded {len(sprites)} asset(s) from '{ASSETS_DIR}'")

    # MediaPipe hand-landmarker setup
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

    # Initialize game state and start the main loop
    state = GameState()
    init_audio()    # open the audio device now so the first slice doesn't hitch
    start = time.monotonic()
    WIN   = "Finger Slicer (Q to quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            # Read a frame from the webcam
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror so the on-screen image acts like a mirror to the player
            frame = cv2.flip(frame, 1)

            # == fingertip detection ===================================
            rgba         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result       = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Extract the index fingertip position (if a hand is detected)
            tip: Optional[tuple[int, int]] = None
            if result.hand_landmarks:
                lm  = result.hand_landmarks[0][INDEX_FINGERTIP]
                tip = (int(lm.x * W), int(lm.y * H))

            # == game step =============================================
            # Snapshot the game-over state before stepping so we can tell when
            # this frame is the one that flips it true
            was_over = state.game_over()

            # Only update the game state if it's not game over
            if not was_over:
                state.update_trail(tip)
                state.maybe_spawn(sprites, W, H)
                state.step(W, H)
                state.detect_slices()
                state.frame_count += 1

            # Play the game-over sound once, on the transition into game over
            if state.game_over() and not was_over:
                play_sound(GAME_OVER_SOUND)

            # == render ================================================
            # Draw projectiles first so they appear behind the blade and HUD
            for p in state.projectiles:
                p.draw(frame)

            # Draw game visual elements on top of projectiles
            draw_miss_marks(frame, state.miss_marks)
            draw_blade(frame, state.trail)
            draw_hud(frame, state, W)

            # If the game is over, draw the game over screen on top of everything else
            if state.game_over():
                draw_game_over(frame, state, W, H)

            # Display the final composited frame
            cv2.imshow(WIN, frame)

            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF

            # Quit on 'q' or Esc key
            if key in (ord('q'), 27):
                break

            # Toggle audio mute on 'm'
            if key == ord('m'):
                muted = toggle_mute()
                print("Audio muted" if muted else "Audio unmuted")

            # Restart the game if it's over and the player presses 'r'
            if state.game_over() and key == ord('r'):
                state.reset()

            # Allow closing the window via [x] button
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
