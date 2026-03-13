from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import cv2
import numpy as np
import pickle
import os
import base64
import time
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from collections import deque

app = Flask(__name__)
CORS(app)

MODELS_DIR = Path("models")
DATA_DIR = Path("data")
MODEL_PATH = Path("model.pkl")
LABELS_PATH = Path("labels.pkl")
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"

os.makedirs(DATA_DIR, exist_ok=True)

# Frame buffer for prediction (30 frames like training)
frame_buffer = deque(maxlen=30)
SEQUENCE_LENGTH = 30
last_prediction_time = 0
PREDICTION_INTERVAL = 2.0  # seconds - same as desktop translator

# All supported signs
ALL_SIGNS = [
    "HELLO", "THANK YOU", "PLEASE", "SORRY", "YES", "NO",
    "GOOD MORNING", "GOOD NIGHT", "I LOVE YOU", "YOU ARE WELCOME", "HELP", "STOP", "CLAP",
    "TEACHER", "SISTER", "BROTHER", "MOTHER", "FATHER", "FRIEND", "FAMILY",
    "COLLEGE", "SCHOOL", "BATHROOM",
    "GO", "WANT", "NEED", "LIKE", "DISLIKE",
    "WATER", "FOOD", "HURT", "HUNGRY",
    "SIGN", "EXCUSE ME", "UNDERSTAND", "HOW ARE YOU", "WHAT", "WHY", "WHERE", "WHO",
    "HAPPY", "SAD", "ANGRY", "TIRED", "BAD", "GOOD",
    "NOW", "LATER", "SOON", "TODAY",
    "SUNDAY", "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY",
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"
]

# Load MediaPipe
print("Loading MediaPipe models...")
base_options = mp_tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
hand_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# Load Pose Landmarker
pose_base_options = mp_tasks.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    min_pose_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
print("MediaPipe models loaded!")

# Try load model
classifier = None
labels = []
try:
    with open(MODEL_PATH, "rb") as f:
        classifier = pickle.load(f)
    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)
    print(f"Loaded model with {len(labels)} signs: {labels}")
except:
    print("No model - will train")

UPPER_BODY_INDICES = [11, 12, 13, 14, 15, 16]

def extract_landmarks(hand_result, pose_result=None):
    """Extract 144 features"""
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)
    
    if hand_result.hand_landmarks and hand_result.handedness:
        for hand_landmarks, handedness_list in zip(hand_result.hand_landmarks, hand_result.handedness):
            label = handedness_list[0].category_name
            coords = []
            for lm in hand_landmarks:
                coords.extend([lm.x, lm.y, lm.z])
            if label == "Right":
                left_hand = np.array(coords[:63])
            elif label == "Left":
                right_hand = np.array(coords[:63])
    
    upper_body = np.zeros(18)
    if pose_result and pose_result.pose_landmarks:
        pose_landmarks = pose_result.pose_landmarks[0]
        coords = []
        for idx in UPPER_BODY_INDICES:
            lm = pose_landmarks[idx]
            coords.extend([lm.x, lm.y, lm.z])
        upper_body = np.array(coords)
    
    return np.concatenate([left_hand, right_hand, upper_body])

def extract_features_from_buffer(buffer_array):
    """Extract 9072 features from 30 frames - matches training exactly"""
    # 1. Flattened raw frames: 30 * 144 = 4320
    flat = buffer_array.flatten()
    # 2. Deltas: 29 * 144 = 4176
    deltas = buffer_array[1:] - buffer_array[:-1]
    flat_deltas = deltas.flatten()
    # 3. Statistics: 4 * 144 = 576
    means = np.mean(buffer_array, axis=0)
    stds = np.std(buffer_array, axis=0)
    mins = np.min(buffer_array, axis=0)
    maxs = np.max(buffer_array, axis=0)
    stats = np.concatenate([means, stds, mins, maxs])
    return np.concatenate([flat, flat_deltas, stats])


def get_sample_counts():
    counts = {}
    for file in DATA_DIR.glob("*.npy"):
        try:
            data = np.load(file, allow_pickle=True)
            if data is not None and hasattr(data, '__len__'):
                counts[file.stem] = len(data)
            else:
                counts[file.stem] = 0
        except:
            counts[file.stem] = 0
    return counts


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    global frame_buffer, last_prediction_time, prediction_history, frames_since_last_hand
    if 'frames_since_last_hand' not in globals():
        frames_since_last_hand = 0
    if 'prediction_history' not in globals():
        prediction_history = deque(maxlen=10)
    
    if classifier is None:
        return jsonify({"detected": False, "error": "Train model first"})
    
    # Handle FormData (from frontend)
    frame_file = request.files.get("frame")
    if frame_file:
        img_bytes = frame_file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif request.is_json:
        data = request.get_json()
        img_data = data.get("image", "")
        if not img_data:
            img_data = data.get("frame", "")
        if "," in img_data:
            img_data = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        return jsonify({"detected": False, "error": "No frame provided"})
    
    if frame is None:
        return jsonify({"detected": False, "error": "No frame"})
    
    try:
        frame = cv2.flip(frame, 1) # Flip horizontally to match collect_data.py
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        hand_result = hand_landmarker.detect(mp_image)
        pose_result = pose_landmarker.detect(mp_image)
    except Exception as e:
        return jsonify({"detected": False, "error": str(e)})
    
    landmarks = extract_landmarks(hand_result, pose_result)
    
    # Check if hand is detected
    hand_detected = hand_result.hand_landmarks is not None and len(hand_result.hand_landmarks) > 0
    
    if hand_detected:
        frames_since_last_hand = 0
        if len(frame_buffer) == 0:
            for _ in range(SEQUENCE_LENGTH - 1):
                frame_buffer.append(landmarks)
            print(f"[INFO] Started new sequence. Hand sum={np.sum(landmarks):.2f}")
        frame_buffer.append(landmarks)
    else:
        frames_since_last_hand += 1
        # If hand is lost for > 5 frames (0.5 seconds), assume they put their hand down entirely
        if frames_since_last_hand > 5:
            if len(frame_buffer) > 0:
                print("[INFO] Hand lost completely, clearing buffer")
                frame_buffer.clear()
            if 'prediction_history' in globals():
                prediction_history.clear()
        elif len(frame_buffer) > 0:
            # For micro-glitches (1-5 frames), duplicate the last known position to keep sequence contiguous
            frame_buffer.append(frame_buffer[-1])
            pass
            
    # Need 30 continuous frames for prediction
    if len(frame_buffer) < SEQUENCE_LENGTH or not hand_detected:
        return jsonify({
            "detected": bool(hand_detected), 
            "prediction": "", 
            "confidence": 0, 
            "predictions": [], 
            "confidences": [],
            "buffer_status": f"{len(frame_buffer)}/{SEQUENCE_LENGTH}",
            "debug_hand_sum": float(np.sum(landmarks) if hand_detected else 0.0)
        })
    
    # Extract features from buffer (9072 features)
    buffer_array = np.array(frame_buffer)
    try:
        features = extract_features_from_buffer(buffer_array).reshape(1, -1)
        
        # Get raw probabilities
        raw_probs = classifier.predict_proba(features)[0]
        
        # Smooth probabilities using a history to prevent wild jumping (10 fps = 10 frames = 1 second)
        prediction_history.append(raw_probs)
        smoothed_probs = np.mean(prediction_history, axis=0)
        
        max_prob = np.max(smoothed_probs)
        pred_idx = np.argmax(smoothed_probs)
        
        # Calculate a pseudo-confidence based on how much the max prob stands out
        confidence = float(max_prob * 100)
        
        top_idx = np.argsort(smoothed_probs)[-3:][::-1]
        
        # If the highest probability is too low, we classify it as 'Unknown'/Low Confidence
        if max_prob < 0.4:
            print(f"[-] Low confidence prediction: {labels[pred_idx]} ({confidence:.1f}%)")
            response = {
                "detected": True,
                "prediction": "",
                "confidence": confidence,
                "predictions": [str(labels[i]) for i in top_idx],
                "confidences": [float(smoothed_probs[i] * 100) for i in top_idx],
                "buffer_status": f"{len(frame_buffer)}/{SEQUENCE_LENGTH}",
                "debug_hand_sum": float(np.sum(landmarks))
            }
        else:
            print(f"[OK] Prediction: {labels[pred_idx]} ({confidence:.1f}%)")
            response = {
                "detected": True,
                "prediction": str(labels[pred_idx]),
                "confidence": confidence,
                "predictions": [str(labels[i]) for i in top_idx],
                "confidences": [float(smoothed_probs[i] * 100) for i in top_idx],
                "buffer_status": f"{len(frame_buffer)}/{SEQUENCE_LENGTH}",
                "debug_hand_sum": float(np.sum(landmarks))
            }

    except Exception as e:
        print(f"[X] Prediction error: {e}")
        response = {"detected": False, "error": str(e)}
    
    return jsonify(response)

@app.route("/add_sign", methods=["POST"])
def add_sign():
    sign_name = request.form["sign_name"].strip().upper()
    frames = request.files.getlist("frames")
    
    if not sign_name or not frames:
        return jsonify({"status": "error", "message": "Name and frames required"})
    
    file_path = DATA_DIR / f"{sign_name}.npy"
    landmarks_list = []
    
    for frame_file in frames:
        try:
            nparr = np.frombuffer(frame_file.read(), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.flip(frame, 1) # Flip horizontally to match collect_data.py
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = hand_landmarker.detect(mp_img)
                lm = extract_landmarks(result)
                if np.sum(lm) > 0:
                    landmarks_list.append(lm)
        except Exception as e:
            print(f"Frame error: {e}")
    
    # Append or create
    if landmarks_list:
        if file_path.exists():
            existing = np.load(file_path, allow_pickle=True)
            combined = list(existing) + landmarks_list
            data = np.array(combined, dtype=object)
        else:
            data = np.array(landmarks_list, dtype=object)
        np.save(file_path, data)
    
    return jsonify({"status": "success", "samples": len(landmarks_list), "sign": sign_name})

@app.route("/retrain", methods=["POST"])
def retrain():
    from sklearn.ensemble import RandomForestClassifier
    
    X_list, y_list = [], []
    print("\n=== Training Data ===")
    
    for file in sorted(DATA_DIR.glob("*.npy")):
        try:
            data = np.load(file, allow_pickle=True)
            if len(data) > 0:
                X_list.append(data)
                y_list.extend([file.stem] * len(data))
                print(f"{file.stem}: {len(data)} samples")
        except Exception as e:
            print(f"Load error {file}: {e}")
    
    if len(X_list) < 2:
        return jsonify({"status": "error", "message": "Need 2+ signs"})
    
    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    print(f"\nTotal: {len(y)} samples, {len(set(y))} signs")
    
    # Features
    n = X.shape[0]
    flat = X.reshape(n, -1)
    deltas = X[:, 1:, :] - X[:, :-1, :]
    flat_d = deltas.reshape(n, -1)
    means = np.mean(X, axis=1)
    stds = np.std(X, axis=1)
    mins = np.min(X, axis=1)
    maxs = np.max(X, axis=1)
    X_feat = np.concatenate([flat, flat_d, means, stds, mins, maxs], axis=1)
    X_feat = np.nan_to_num(X_feat)
    
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_feat, y)
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(list(clf.classes_), f)
    
    global classifier, labels
    classifier = clf
    labels = list(clf.classes_)
    
    acc = clf.score(X_feat, y)
    return jsonify({"status": "success", "signs": len(labels), "samples": len(y), "accuracy": acc*100})

@app.route("/samples")
def samples():
    return jsonify(get_sample_counts())

@app.route("/debug")
def debug():
    c = get_sample_counts()
    print("\n=== DEBUG ===")
    for k, v in sorted(c.items()): print(f"{k}: {v}")
    return jsonify(c)

if __name__ == "__main__":
    print("="*50)
    print("ASL Translator - Enhanced")
    print(f"Signs: {len(ALL_SIGNS)}")
    print(f"Data: {get_sample_counts()}")
    print("http://localhost:5001")
    print("="*50)
    app.run(debug=True, port=5001, host='0.0.0.0')
