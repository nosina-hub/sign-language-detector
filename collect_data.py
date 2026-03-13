"""
Data collection script for ASL sign language translator.

Uses OpenCV and the MediaPipe Tasks API (v0.10.32+) to capture hand and upper
body landmark sequences for ASL signs. Landmarks are saved as numpy arrays
for training.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SIGNS = [
    "YES", "NO", "THANK YOU", "HELLO", "I LOVE YOU",
    "GOODBYE", "YOU ARE WELCOME", "PLEASE", "SORRY", "CLAP",
    "GOOD MORNING", "GOOD NIGHT", "GOOD JOB", "GREAT",
    "MONEY", "FRIEND", "FAMILY", "HELP", "EAT", "DRINK",
    "BATHROOM", "WORK", "HOME", "SCHOOL", "WATER", "FOOD",
    "TIME", "WHAT", "WHY", "HOW", "NAME", "YOUR", "MY",
    "MORE", "LESS", "AGAIN", "STOP", "GO", "WAIT", "WANT",
    "NEED", "LIKE", "DISLIKE", "HAPPY", "SAD", "ANGRY",
    "TIRED", "HUNGRY", "THIRSTY", "SICK", "OKAY", "PERFECT",
]
SEQUENCE_LENGTH = 30          # frames per sample (~1 second at 30 fps)
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Feature vector layout per frame:
#   left hand  : 21 landmarks * 3 (x,y,z) = 63
#   right hand : 21 landmarks * 3 (x,y,z) = 63
#   upper body : 6 landmarks  * 3 (x,y,z) = 18   (shoulders, elbows, wrists)
# Total = 144
NUM_HAND_LANDMARKS = 21
NUM_HAND_FEATURES = NUM_HAND_LANDMARKS * 3        # 63
NUM_POSE_LANDMARKS = 6
NUM_POSE_FEATURES = NUM_POSE_LANDMARKS * 3         # 18
TOTAL_FEATURES = NUM_HAND_FEATURES * 2 + NUM_POSE_FEATURES  # 144

# MediaPipe Pose landmark indices for upper body (shoulders, elbows, wrists)
POSE_UPPER_BODY_INDICES = [
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
]

# Hand connections for manual drawing (since mp.solutions.drawing_utils is removed)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]

# Pose connections for upper body drawing
POSE_UPPER_BODY_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13),  # left shoulder -> left elbow
    (13, 15),  # left elbow -> left wrist
    (12, 14),  # right shoulder -> right elbow
    (14, 16),  # right elbow -> right wrist
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_hand_landmarks(hand_landmark_list):
    """Extract (x, y, z) for all 21 hand landmarks into a flat array of 63."""
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmark_list]
    ).flatten()


def extract_pose_landmarks(pose_landmark_list):
    """Extract upper-body pose landmarks (shoulders, elbows, wrists) -> 18 values."""
    values = []
    for idx in POSE_UPPER_BODY_INDICES:
        lm = pose_landmark_list[idx]
        values.extend([lm.x, lm.y, lm.z])
    return np.array(values)


def build_feature_vector(hand_result, pose_result):
    """
    Build a single feature vector (144,) from MediaPipe Tasks results.

    Layout: [left_hand(63) | right_hand(63) | pose(18)]
    Pads with zeros when a hand or pose is not detected.
    """
    left_hand = np.zeros(NUM_HAND_FEATURES)
    right_hand = np.zeros(NUM_HAND_FEATURES)
    pose = np.zeros(NUM_POSE_FEATURES)

    if hand_result.hand_landmarks and hand_result.handedness:
        for hand_landmarks, handedness in zip(
            hand_result.hand_landmarks,
            hand_result.handedness,
        ):
            label = handedness[0].category_name  # "Left" or "Right"
            features = extract_hand_landmarks(hand_landmarks)
            # MediaPipe labels are mirrored (camera view), so we swap.
            if label == "Right":
                left_hand = features
            else:
                right_hand = features

    if pose_result.pose_landmarks:
        pose = extract_pose_landmarks(pose_result.pose_landmarks[0])

    return np.concatenate([left_hand, right_hand, pose])


def draw_hand_landmarks(frame, hand_landmark_list):
    """Draw hand landmarks and connections on the frame using OpenCV."""
    h, w, _ = frame.shape
    points = []
    for lm in hand_landmark_list:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)


def draw_pose_landmarks(frame, pose_landmark_list):
    """Draw upper-body pose landmarks and connections on the frame using OpenCV."""
    h, w, _ = frame.shape
    # Build a dict of index -> pixel position for the landmarks we care about
    landmark_points = {}
    for idx in POSE_UPPER_BODY_INDICES:
        lm = pose_landmark_list[idx]
        px, py = int(lm.x * w), int(lm.y * h)
        landmark_points[idx] = (px, py)
        cv2.circle(frame, (px, py), 5, (255, 0, 0), -1)

    for start_idx, end_idx in POSE_UPPER_BODY_CONNECTIONS:
        if start_idx in landmark_points and end_idx in landmark_points:
            cv2.line(frame, landmark_points[start_idx], landmark_points[end_idx], (255, 0, 0), 2)


def load_existing_data(sign_name):
    """Load existing .npy file for a sign, or return None."""
    path = os.path.join(DATA_DIR, f"{sign_name}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def save_data(sign_name, data):
    """Save numpy array to data/{sign_name}.npy."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{sign_name}.npy")
    np.save(path, data)


def draw_countdown(frame, number, sign_name):
    """Draw a large countdown number centered on the frame."""
    h, w, _ = frame.shape
    cv2.putText(
        frame, f"Collecting: {sign_name}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )
    text = str(number)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 5
    thickness = 8
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 255), thickness)


def draw_recording(frame, current_frame, total_frames, sign_name):
    """Draw recording indicator on the frame."""
    cv2.putText(
        frame, f"Collecting: {sign_name}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )
    cv2.putText(
        frame, f"RECORDING  {current_frame}/{total_frames}",
        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
    )
    bar_w = 300
    bar_h = 20
    x, y = 20, 85
    progress = int(bar_w * current_frame / total_frames)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (100, 100, 100), -1)
    cv2.rectangle(frame, (x, y), (x + progress, y + bar_h), (0, 0, 255), -1)


def draw_saved_message(frame, sign_name, count):
    """Draw the 'Saved!' confirmation on the frame."""
    cv2.putText(
        frame, f"Saved! Total samples for {sign_name}: {count}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,
    )
    cv2.putText(
        frame, "Press any key to collect another sign...",
        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def draw_sign_menu(frame, selected_idx):
    """Draw the sign selection menu on the frame with keyboard navigation."""
    h, w, _ = frame.shape
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(
        frame, "W/S or I/K: navigate | P/L: jump 10 | ENTER: select | Q: quit",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    
    start_idx = max(0, selected_idx - 10)
    end_idx = min(len(SIGNS), start_idx + 15)
    
    for i in range(start_idx, end_idx):
        y_pos = 75 + (i - start_idx) * 35
        if i == selected_idx:
            color = (0, 255, 0)
            cv2.rectangle(frame, (15, y_pos - 25), (w - 20, y_pos + 5), (50, 100, 50), -1)
        else:
            color = (200, 200, 200)
        cv2.putText(
            frame, f"  {i+1:2d}. {SIGNS[i]}",
            (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
        )


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Download model files if they don't exist ---
    MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

    hand_model_path = os.path.join(MODELS_DIR, "hand_landmarker.task")
    pose_model_path = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")

    for url, path in [(HAND_MODEL_URL, hand_model_path), (POSE_MODEL_URL, pose_model_path)]:
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)}...")
            urllib.request.urlretrieve(url, path)
            print(f"  Downloaded to {path}")

    # --- Create MediaPipe Tasks landmarkers ---
    hand_base_options = python.BaseOptions(model_asset_path=hand_model_path)
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    pose_base_options = python.BaseOptions(model_asset_path=pose_model_path)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    print("\nASL Data Collection")
    print("Use W/S or I/K to navigate, P/L to jump 10, ENTER to select.")
    print("Or use number keys 1-9, 0. Press Q to quit.\n")

    # Use actual timestamps to avoid "must be monotonically increasing" error
    import time as time_module
    base_timestamp = int(time_module.time() * 1000)

    base_timestamp = int(time.time() * 1000)

    while True:
        # States: "menu", "countdown", "recording", "saved"
        state = "menu"
        selected_sign_idx = 0
        countdown_start = None
        sequence = []
        saved_time = None
        saved_count = 0
        sign_name = ""
        
        frame_count = 0

        while cap.isOpened() and state != "quit":
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to read frame.")
                break

            frame_count += 1
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Wrap frame for MediaPipe Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Use frame count for timestamps (monotonically increasing)
            frame_timestamp_ms = frame_count * 33  # ~30 fps

            hand_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # Draw hand landmarks
            if hand_result.hand_landmarks:
                for hand_lms in hand_result.hand_landmarks:
                    draw_hand_landmarks(frame, hand_lms)

            # Draw pose landmarks (upper body only)
            if pose_result.pose_landmarks:
                draw_pose_landmarks(frame, pose_result.pose_landmarks[0])

            # --- State machine ---

            if state == "menu":
                draw_sign_menu(frame, selected_sign_idx)
                
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    hand_landmarker.close()
                    pose_landmarker.close()
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\nDone. Data saved in:", DATA_DIR)
                    return
                elif key == 13:  # ENTER
                    sign_name = SIGNS[selected_sign_idx]
                    existing = load_existing_data(sign_name)
                    existing_count = existing.shape[0] if existing is not None else 0
                    print(f"Selected: {sign_name} (current samples: {existing_count})")
                    countdown_start = time.time()
                    state = "countdown"
                elif key in [ord("w"), ord("W"), ord("i"), ord("I")]:  # UP
                    selected_sign_idx = (selected_sign_idx - 1) % len(SIGNS)
                elif key in [ord("s"), ord("S"), ord("k"), ord("K")]:  # DOWN
                    selected_sign_idx = (selected_sign_idx + 1) % len(SIGNS)
                elif key in [ord("p"), ord("P")]:  # Page up - jump up 10
                    selected_sign_idx = (selected_sign_idx - 10) % len(SIGNS)
                elif key in [ord("l"), ord("L")]:  # Page down - jump down 10
                    selected_sign_idx = (selected_sign_idx + 10) % len(SIGNS)
                elif ord("1") <= key <= ord("9"):
                    selected_sign_idx = key - ord("1")
                elif key == ord("0"):
                    selected_sign_idx = 9

            elif state == "countdown":
                elapsed = time.time() - countdown_start
                remaining = 3 - int(elapsed)
                if remaining > 0:
                    draw_countdown(frame, remaining, sign_name)
                else:
                    state = "recording"
                    sequence = []

            elif state == "recording":
                features = build_feature_vector(hand_result, pose_result)
                sequence.append(features)
                draw_recording(frame, len(sequence), SEQUENCE_LENGTH, sign_name)

                if len(sequence) >= SEQUENCE_LENGTH:
                    new_sample = np.array(sequence).reshape(1, SEQUENCE_LENGTH, TOTAL_FEATURES)

                    existing = load_existing_data(sign_name)
                    if existing is not None and existing.shape[1] == SEQUENCE_LENGTH:
                        combined = np.concatenate([existing, new_sample], axis=0)
                    else:
                        if existing is not None:
                            print(f"  Note: existing data has {existing.shape[1]} frames, new has {SEQUENCE_LENGTH}")
                        combined = new_sample

                    save_data(sign_name, combined)
                    saved_count = combined.shape[0]
                    saved_time = time.time()
                    state = "saved"
                    print(f"  Saved sample for '{sign_name}' — total: {saved_count}")

            elif state == "saved":
                draw_saved_message(frame, sign_name, saved_count)
                if time.time() - saved_time > 1.5:
                    state = "menu"

            cv2.imshow("ASL Data Collection", frame)

            key = cv2.waitKey(10) & 0xFF  # Single consistent waitKey
            if key == ord("q"):
                hand_landmarker.close()
                pose_landmarker.close()
                cap.release()
                cv2.destroyAllWindows()
                print("\nDone. Data saved in:", DATA_DIR)
                return

        print("Webcam closed - exiting")

    hand_landmarker.close()
    pose_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone. Data saved in:", DATA_DIR)


if __name__ == "__main__":
    main()
