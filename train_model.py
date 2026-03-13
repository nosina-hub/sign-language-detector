"""
Train a sign language classifier on landmark sequence data collected by collect_data.py.

Loads .npy files from data/, engineers features from landmark sequences,
trains a RandomForestClassifier, and saves the model and label list.
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.pkl")

FRAMES_PER_SEQUENCE = 30
FEATURES_PER_FRAME = 144  # 63 (left hand) + 63 (right hand) + 18 (upper body pose)


def load_data(data_dir):
    """Load all .npy files from the data directory and create labels."""
    X_list = []
    y_list = []
    label_names = []

    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' does not exist.")
        print("Please run collect_data.py first to record sign language samples.")
        return None, None, None

    npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])

    if len(npy_files) == 0:
        print("No .npy data files found in the data/ directory.")
        print("Please run collect_data.py first to record sign language samples.")
        return None, None, None

    print(f"Found {len(npy_files)} sign(s) in data directory:\n")

    for npy_file in npy_files:
        sign_name = npy_file.replace(".npy", "")
        filepath = os.path.join(data_dir, npy_file)
        data = np.load(filepath)  # shape: (num_samples, 30, 144)

        num_samples = data.shape[0]
        print(f"  {sign_name}: {num_samples} samples")

        if num_samples < 20:
            print(f"    WARNING: Very few samples for '{sign_name}'. "
                  f"Collect 20-30 samples for better accuracy.")

        X_list.append(data)
        y_list.extend([sign_name] * num_samples)
        if sign_name not in label_names:
            label_names.append(sign_name)

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)

    print(f"\nTotal samples: {len(y)}")
    print(f"Labels: {label_names}\n")

    return X, y, label_names


def engineer_features(X):
    """
    Engineer features from raw landmark sequences.

    For each sample (30 frames x 144 features):
      1. Flatten all frames -> 30 * 144 = 4320 features (full motion capture)
      2. Delta features: difference between consecutive frames (29 * 144 = 4176 features, flattened)
      3. Per-landmark statistics over time: mean, std, min, max (4 * 144 = 576 features)

    All concatenated into one feature vector per sample.
    """
    num_samples = X.shape[0]

    # 1. Flatten all frames
    flat = X.reshape(num_samples, -1)  # (N, 4320)

    # 2. Delta features: differences between consecutive frames
    deltas = X[:, 1:, :] - X[:, :-1, :]  # (N, 29, 144)
    flat_deltas = deltas.reshape(num_samples, -1)  # (N, 4176)

    # 3. Per-landmark statistics over time (across the 30 frames)
    means = np.mean(X, axis=1)  # (N, 144)
    stds = np.std(X, axis=1)    # (N, 144)
    mins = np.min(X, axis=1)    # (N, 144)
    maxs = np.max(X, axis=1)    # (N, 144)

    # Concatenate all features
    features = np.concatenate([flat, flat_deltas, means, stds, mins, maxs], axis=1)

    print(f"Feature vector size per sample: {features.shape[1]}")
    print(f"  Flattened frames: {flat.shape[1]}")
    print(f"  Delta features:   {flat_deltas.shape[1]}")
    print(f"  Statistics:       {means.shape[1] * 4} (mean + std + min + max)\n")

    return features


def main():
    print("=" * 60)
    print("Sign Language Translator - Model Training")
    print("=" * 60)
    print()

    # Load data
    X_raw, y, label_names = load_data(DATA_DIR)

    if X_raw is None:
        return

    total_samples = len(y)
    num_classes = len(label_names)

    if total_samples < 10:
        print("Not enough data to train a reliable model.")
        print("Please collect more samples using collect_data.py.")
        print("Aim for at least 20-30 samples per sign.")
        return

    if num_classes < 2:
        print("Need at least 2 different signs to train a classifier.")
        print("Please collect data for more signs using collect_data.py.")
        return

    # Engineer features
    print("Engineering features...")
    X_features = engineer_features(X_raw)

    # Replace any NaN or inf values with 0
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Calculate minimum samples per class
    from collections import Counter
    min_per_class = min(Counter(y).values())
    
    # Determine test size based on data size
    if min_per_class < 3:
        test_size = 1  # Use single sample for test if very limited
        use_stratify = False
    elif total_samples < 50:
        test_size = 0.15  # 15% for small datasets
        use_stratify = True
    else:
        test_size = 0.2  # 20% for larger datasets
        use_stratify = True

    # Split into train and test sets
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42
        )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set:     {len(X_test)} samples\n")

    # Train RandomForestClassifier
    print("Training RandomForestClassifier (n_estimators=200)...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Training complete.\n")

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred))
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)\n")

    if accuracy < 0.7:
        print("Accuracy is below 70%. Consider collecting more data per sign")
        print("(aim for 30+ samples each) to improve performance.\n")

    # Save model and labels
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {MODEL_PATH}")

    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_names, f)
    print(f"Labels saved to {LABELS_PATH}")

    print()
    print("Done! You can now use the trained model for real-time translation.")


if __name__ == "__main__":
    main()
