import os
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def load_samples(data_dir='models/quadrants'):
    """Load samples from JSON files."""
    samples = []
    labels = []
    label_map = {'top_left': 0, 'top_right': 1, 'bottom_left': 2, 'bottom_right': 3}

    for quadrant, label in label_map.items():
        json_path = os.path.join(data_dir, f"{quadrant}_data.json")
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} does not exist. Skipping.")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)
            for sample in data:
                left_rel = sample['left_eye']
                right_rel = sample['right_eye']
                features = left_rel + right_rel  # Concatenate left and right eye relative positions
                samples.append(features)
                labels.append(label)

    return np.array(samples), np.array(labels)

def train_model(output_path='models/look_at_quadrants_model.pkl', scaler_path='models/scaler.pkl'):
    """Train an SVM classifier based on relative pupil positions."""
    # Load samples and labels
    X, y = load_samples()
    if X.size == 0 or y.size == 0:
        print("No data found. Please capture images first.")
        return

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # Save the scaler for later use
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    # Train an SVM classifier with an RBF kernel
    model = SVC(kernel='rbf', probability=True, gamma='auto', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['top_left', 'top_right', 'bottom_left', 'bottom_right']))

    # Save the trained model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train_model()
