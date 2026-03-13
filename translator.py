"""
Real-time ASL Sign Language Translator

Uses webcam, MediaPipe Tasks API, and a trained model to recognize ASL signs and speak them.
Signs are defined in collect_data.py and loaded from the trained model.
"""

import os
import pickle
import time
import threading
import urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3


# ---------------------------------------------------------------------------
# Sample count helper
# ---------------------------------------------------------------------------

def get_sample_counts():
    """Dynamically read sample counts from data folder"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    counts = {}
    for file in os.listdir(data_dir):
        if file.endswith('.npy'):
            sign_name = file.replace('.npy', '')
            try:
                data = np.load(os.path.join(data_dir, file), allow_pickle=True)
                counts[sign_name] = len(data) if data is not None else 0
            except:
                counts[sign_name] = 0
    return counts


def save_sample(sign_name, landmarks):
    """Save a single sample to the data folder - appends to existing"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, f"{sign_name}.npy")
    new_sample = np.array([landmarks], dtype=object)
    
    if os.path.exists(file_path):
        existing = np.load(file_path, allow_pickle=True)
        updated = np.concatenate([existing, new_sample])
    else:
        updated = new_sample
    
    np.save(file_path, updated)
    return len(updated)


# ---------------------------------------------------------------------------
# Feature extraction (must match train_model.py exactly)
# ---------------------------------------------------------------------------

def extract_features(frames: np.ndarray) -> np.ndarray:
    """
    Given a (30, 144) array of landmark data, produce the feature vector
    used by the trained model.

    Features:
        1. Flattened raw frames  -> 30 * 144 = 4320
        2. Deltas between consecutive frames (frames[1:] - frames[:-1])
           flattened -> 29 * 144 = 4176
        3. Per-landmark statistics across frames:
           mean, std, min, max for each of the 144 landmarks
           -> 144 * 4 = 576

    Total: 4320 + 4176 + 576 = 9072
    """
    # 1. Flattened raw frames
    flat = frames.flatten()

    # 2. Deltas between consecutive frames
    deltas = frames[1:] - frames[:-1]
    flat_deltas = deltas.flatten()

    # 3. Per-landmark statistics (across the time axis)
    means = np.mean(frames, axis=0)
    stds = np.std(frames, axis=0)
    mins = np.min(frames, axis=0)
    maxs = np.max(frames, axis=0)
    stats = np.concatenate([means, stds, mins, maxs])

    return np.concatenate([flat, flat_deltas, stats])


# ---------------------------------------------------------------------------
# Text-to-speech helper (runs in a background thread)
# ---------------------------------------------------------------------------

class Speaker:
    """Non-blocking TTS wrapper using pyttsx3 in a dedicated thread."""

    def __init__(self):
        self._queue: deque = deque()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        while True:
            if self._queue:
                text = self._queue.popleft()
                engine.say(text)
                engine.runAndWait()
            else:
                time.sleep(0.05)

    def speak(self, text: str):
        self._queue.append(text)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

# Hand landmark connections for manual drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Upper-body pose landmark indices (same as collect_data.py):
# shoulders (11, 12), elbows (13, 14), wrists (15, 16)
UPPER_BODY_INDICES = [11, 12, 13, 14, 15, 16]

# Upper-body pose connections for manual drawing
# Only draw connections between the upper-body landmarks we use
POSE_UPPER_CONNECTIONS = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 12),            # shoulders
]


def draw_hand_landmarks(image, hand_landmarks_list):
    """Draw hand landmarks and connections on the image using OpenCV."""
    h, w, _ = image.shape
    for hand_landmarks in hand_landmarks_list:
        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw landmark points
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)


def draw_pose_landmarks(image, pose_landmarks_list):
    """Draw upper-body pose landmarks and connections on the image using OpenCV."""
    h, w, _ = image.shape
    for pose_landmarks in pose_landmarks_list:
        # Draw connections between upper-body landmarks
        for start_idx, end_idx in POSE_UPPER_CONNECTIONS:
            start = pose_landmarks[start_idx]
            end = pose_landmarks[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(image, (x1, y1), (x2, y2), (255, 200, 0), 2)
        # Draw upper-body landmark points
        for idx in UPPER_BODY_INDICES:
            lm = pose_landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (255, 0, 255), -1)


def draw_overlay(image, text, confidence, sentence, fps, sample_feedback=""):
    """Draw UI elements on the camera frame."""
    h, w, _ = image.shape

    # --- Top bar (detected sign) ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    if text:
        label = f"{text}  ({confidence:.0%})"
        cv2.putText(image, label, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 128), 4,
                    cv2.LINE_AA)

    # --- Sample feedback (when A is pressed) ---
    if sample_feedback:
        cv2.putText(image, sample_feedback, (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
                    cv2.LINE_AA)

    # --- Bottom sentence bar ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - 120), (w, h), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    sentence_text = " ".join(sentence) if sentence else "(no signs detected yet)"
    cv2.putText(image, sentence_text, (20, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                cv2.LINE_AA)

    # --- Instructions ---
    instructions = "SPACE: speak | C: clear | A: add sample | Q: quit"
    cv2.putText(image, instructions, (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1,
                cv2.LINE_AA)

    # --- FPS ---
    cv2.putText(image, f"FPS: {fps:.0f}", (w - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
                cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------

def extract_landmarks(hand_result, pose_result) -> np.ndarray:
    """
    Extract a single frame of landmarks from the new MediaPipe Tasks API results.

    Returns a (144,) array:
        - 63 values for left hand  (21 landmarks * 3 coords)
        - 63 values for right hand (21 landmarks * 3 coords)
        - 18 values for upper body pose (6 landmarks * 3 coords)
    """
    # Left hand (63)
    left_hand = np.zeros(63)
    # Right hand (63)
    right_hand = np.zeros(63)

    if hand_result.hand_landmarks and hand_result.handedness:
        for hand_landmarks, handedness_list in zip(
            hand_result.hand_landmarks,
            hand_result.handedness
        ):
            # handedness_list is a list of Category; take the first one
            label = handedness_list[0].category_name
            coords = []
            for lm in hand_landmarks:
                coords.extend([lm.x, lm.y, lm.z])
            # Swap labels to match collect_data.py convention:
            # Because the frame is flipped, MediaPipe's "Right" = user's right hand
            # but in collect_data.py "Right" was mapped to left_hand array.
            if label == "Right":
                left_hand = np.array(coords[:63])
            elif label == "Left":
                right_hand = np.array(coords[:63])

    # Upper body pose (18)
    upper_body = np.zeros(18)
    if pose_result.pose_landmarks:
        # pose_landmarks is a list of lists; take the first detected pose
        pose_landmarks = pose_result.pose_landmarks[0]
        coords = []
        for idx in UPPER_BODY_INDICES:
            lm = pose_landmarks[idx]
            coords.extend([lm.x, lm.y, lm.z])
        upper_body = np.array(coords)

    return np.concatenate([left_hand, right_hand, upper_body])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    # --- Auto-download model files if needed ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
    POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")

    HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

    for url, path in [(HAND_MODEL_URL, HAND_MODEL_PATH), (POSE_MODEL_URL, POSE_MODEL_PATH)]:
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)}...")
            urllib.request.urlretrieve(url, path)

    # Load model and labels
    model_pkl_path = os.path.join(SCRIPT_DIR, "model.pkl")
    labels_pkl_path = os.path.join(SCRIPT_DIR, "labels.pkl")

    with open(model_pkl_path, "rb") as f:
        model = pickle.load(f)
    with open(labels_pkl_path, "rb") as f:
        labels = pickle.load(f)

    # Print signs with correct numbering from labels.pkl
    print(f"\n=== Signs in Model ({len(labels)} total) ===")
    sample_counts = get_sample_counts()
    for idx, label in enumerate(labels):
        count = sample_counts.get(label, 0)
        print(f"  {idx}: {label} ({count} samples)")
    print("======================================\n")

    # Show sample counts for all signs in data folder
    print("=== All Data Samples ===")
    for sign, count in sorted(sample_counts.items()):
        print(f"  {sign}: {count} samples")
    print("======================================\n")

    # --- MediaPipe Tasks API setup ---
    # Hand Landmarker
    hand_base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    # Pose Landmarker
    pose_base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    # Speaker
    speaker = Speaker()

    # Webcam - try different indices if 0 doesn't work
    for cam_idx in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            print(f"Webcam opened on index {cam_idx}")
            break
    else:
        print("ERROR: Could not open any webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State
    BUFFER_SIZE = 30
    frame_buffer: deque = deque(maxlen=BUFFER_SIZE)
    frame_count = 0
    current_sign = ""
    current_confidence = 0.0
    last_sign = ""
    last_sign_time = 0.0
    last_prediction_time = 0.0  # For 2-second prediction interval
    sentence: list[str] = []
    prev_time = time.time()
    fps = 0.0

    CONFIDENCE_THRESHOLD = 0.25  # Lowered for better detection
    REPEAT_COOLDOWN = 1.5  # seconds before same sign can be added again

    print("Starting translator. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Frame read failed, retrying...")
            time.sleep(0.3)
            cap.release()
            for cam_idx in [0, 1, 2]:
                cap = cv2.VideoCapture(cam_idx)
                if cap.isOpened():
                    print(f"Reopened webcam on index {cam_idx}")
                    break
            else:
                print("ERROR: Could not reopen webcam.")
                break
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image and compute timestamp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        # Run MediaPipe Tasks
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw landmarks on frame using OpenCV
        if hand_result.hand_landmarks:
            draw_hand_landmarks(frame, hand_result.hand_landmarks)
        if pose_result.pose_landmarks:
            draw_pose_landmarks(frame, pose_result.pose_landmarks)

        # Extract landmarks and add to buffer
        landmarks = extract_landmarks(hand_result, pose_result)
        frame_buffer.append(landmarks)
        frame_count += 1
        
        # Check if hand is detected
        hand_detected = hand_result.hand_landmarks is not None
        
        # Predict every 2 seconds (not every frame) - STABILIZED
        PREDICTION_INTERVAL = 2.0  # seconds
        should_predict = (
            len(frame_buffer) == BUFFER_SIZE and 
            (time.time() - last_prediction_time) >= PREDICTION_INTERVAL
        )
        
        if should_predict and hand_detected:
            buffer_array = np.array(frame_buffer)  # (30, 144)
            features = extract_features(buffer_array).reshape(1, -1)

            probabilities = model.predict_proba(features)[0]
            best_idx = np.argmax(probabilities)
            confidence = probabilities[best_idx]
            predicted_label = labels[best_idx]
            last_prediction_time = time.time()

            if confidence > CONFIDENCE_THRESHOLD:
                current_sign = predicted_label
                current_confidence = confidence
                now = time.time()

                # Accept if different sign or cooldown elapsed
                if predicted_label != last_sign or (now - last_sign_time) >= REPEAT_COOLDOWN:
                    sentence.append(predicted_label)
                    speaker.speak(predicted_label)
                    last_sign = predicted_label
                    last_sign_time = now
            else:
                # Keep showing last prediction with lower confidence
                pass

        sample_feedback = ""
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord(" "):
            if sentence:
                full = " ".join(sentence)
                print(f"Speaking sentence: {full}")
                speaker.speak(full)
        elif key == ord("c") or key == ord("C"):
            sentence.clear()
            current_sign = ""
            current_confidence = 0.0
            last_sign = ""
            print("Sentence cleared.")
        elif key == ord("a") or key == ord("A"):
            # Press A to save current frame as sample for predicted sign
            if hand_result.hand_landmarks and current_sign:
                total = save_sample(current_sign, landmarks)
                sample_feedback = f"Sample Added for {current_sign} (total: {total})"
                print(f"Saved sample for {current_sign} (total: {total})")
                sample_counts = get_sample_counts()  # Refresh counts
            elif hand_result.hand_landmarks:
                sample_feedback = "No sign detected - show a sign first"
                print("Cannot add sample: no sign detected")
            else:
                sample_feedback = "No hand detected"
                print("Cannot add sample: no hand detected")
        
        # FPS calculation
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # Draw UI (after key handling so sample_feedback is updated)
        draw_overlay(frame, current_sign, current_confidence, sentence, fps, sample_feedback)

        cv2.imshow("ASL Translator", frame)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    pose_landmarker.close()
    print("Translator stopped.")


if __name__ == "__main__":
    main()
