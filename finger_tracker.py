from   config import *
from   utils  import get_device

import time
import urllib.request
from   pathlib                import Path

import cv2
import mediapipe              as mp
from   mediapipe.tasks        import python as mp_python
from   mediapipe.tasks.python import vision as mp_vision

def load_model() -> Path:
    """
    Download the hand landmark model on first run and return its cached path.

    Returns:
        Path: Absolute path to the local `hand_landmarker.task` file.
    """
    if not HAND_MODEL_PATH.exists():
        # Create directory if it doesn't exist
        HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand_landmarker model to {HAND_MODEL_PATH} ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    return HAND_MODEL_PATH

def use_device():
    """
    Map the torch device returned by `get_device()` to a MediaPipe delegate.
    CUDA / MPS -> GPU, anything else -> CPU.

    Returns:
        mp_python.BaseOptions.Delegate: The MediaPipe inference backend to use (`GPU` if CUDA/MPS is available, otherwise `CPU`).
    """
    device = get_device()
    if device.type in ("cuda", "mps"):
        print(f"Torch reports '{device.type}': requesting MediaPipe GPU delegate.")
        return mp_python.BaseOptions.Delegate.GPU
    print(f"Torch reports '{device.type}': using MediaPipe CPU delegate.")
    return mp_python.BaseOptions.Delegate.CPU

def detect_tracking_issues(result) -> list[str]:
    """
    Inspect a `HandLandmarkerResult` and report what looks wrong with the
    current detection.

    Args:
        result: The `HandLandmarkerResult` returned by `landmarker.detect_for_video()` for the current frame.

    Returns:
        list[str]: Human-readable warning messages
        (e.g. "No hand detected", "Hand partially out of view (3/21 pts)").
        An empty list means everything looks fine.
    """
    # No hand detected at all in the frame
    if not result.hand_landmarks:
        return ["No hand detected"]

    issues: list[str] = []
    for i, landmarks in enumerate(result.hand_landmarks):
        # Landmarks outside normalized [0, 1] coords => hand is partially off-screen
        out_of_frame = sum(
            1 for lm in landmarks
            if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0)
        )
        if out_of_frame:
            issues.append(f"Hand partially out of view ({out_of_frame}/21 pts)")

        # Low handedness score => the classifier isn't sure this is really a hand
        if result.handedness and i < len(result.handedness):
            score = result.handedness[i][0].score
            if score < HAND_HANDEDNESS_WARN:
                issues.append(f"Low tracking confidence ({score:.2f})")

    return issues


def draw_warnings(frame, messages: list[str]) -> None:
    """
    Stack warning messages in the top-left corner of the frame, rendered in
    red with a black outline for readability. Modifies `frame` in place.

    Args:
        frame: BGR image (`numpy.ndarray`) the warnings will be drawn onto.
        messages: List of warning strings to render, one per line.
    """
    for i, msg in enumerate(messages):
        y = 30 + i * 28
        # Black outline first, then red text on top: makes the warning readable regardless of what's behind it in the frame
        cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


def draw_hand(frame, landmarks):
    """
    Draw the hand skeleton, joints and the highlighted index fingertip on
    `frame` in place.

    Args:
        frame: BGR image (`numpy.ndarray`) the hand overlay will be drawn onto.
        landmarks: Sequence of 21 normalized landmarks
        (each with `x` and `y` in [0, 1]) for a single detected hand.
    """
    # MediaPipe returns normalized coordinates in [0, 1]
    # scale them back to pixel space using the actual frame size before drawing
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Skeleton: connect landmark pairs with white lines
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    # Joints: small red dot on each of the 21 landmarks
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    # Emphasize the index fingertip with a green ring + its pixel coordinates
    tx, ty = pts[INDEX_FINGERTIP]
    cv2.circle(frame, (tx, ty), 12, (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"({tx}, {ty})",            # string to display
        (tx + 15, ty - 10),         # bottom-left corner of the text
        cv2.FONT_HERSHEY_SIMPLEX,   # font
        0.6,                        # font scale
        (0, 255, 0),                # color (green)
        2,                          # thickness
    )

def main() -> None:
    """
    Open the default webcam and run the real-time hand-tracking loop until the user presses 'q'.

    Returns:
        None
    """

    # Load model
    model_path = load_model()
    
    options = mp_vision.HandLandmarkerOptions(
            # Tells MediaPipe which .task (trained neural network weights) to load
            base_options=mp_python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=use_device(),
            ),
            # MediaPipe running mode  
            running_mode=mp_vision.RunningMode.VIDEO,
            # Upper bound on how many hands the model will return per frame
            num_hands=1,
            # On new detection (no prior hand to track), it only reports a hand if its confidence score is above the threshold
            min_hand_detection_confidence=HAND_DETECT_CONFIDENCE,
            # Once a hand is found, the tracker keeps following it across frames as long as its tracking confidence is above the threshold
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
    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam mode: {actual_w}x{actual_h} @ {actual_fps:.1f} fps "
          f"(requested {CAM_WIDTH}x{CAM_HEIGHT} @ {CAM_FPS} fps)")

    # Reference point for the timestamps we feed to detect_for_video()
    start = time.monotonic()
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror horizontally so the preview behaves like a mirror
            # moving your hand right makes it move right on screen
            frame = cv2.flip(frame, 1)
            # OpenCV captures BGR; MediaPipe expects RGB(A)
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
            timestamp_ms = int((time.monotonic() - start) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            # result.hand_landmarks is a list with one entry per detected hand
            if result.hand_landmarks:
                for landmarks in result.hand_landmarks:
                    draw_hand(frame, landmarks)

            # Overlay any tracking warnings
            issues = detect_tracking_issues(result)
            if issues:
                draw_warnings(frame, issues)

            cv2.imshow("Finger Tracker (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()