import json
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


window = 180
step = 60

def count_yawns(mar_values):
    yawn_count = 0
    yawn_in_progress = False
    yawn_frames = 0
    for mar in mar_values:
        if mar > 0.6:
            if not yawn_in_progress:
                yawn_in_progress = True
                yawn_frames = 1
            else:
                yawn_frames += 1
        else:
            if yawn_in_progress and yawn_frames >= 45:
                yawn_count += 1
            yawn_in_progress = False
            yawn_frames = 0
    return yawn_count

def extract_features(ear, mar):
    perclos = sum(1 for e in ear if e < 0.2) / len(ear)
    return [
        np.mean(ear),
        np.min(ear),
        np.std(ear),
        perclos,
        np.mean(mar),
        np.max(mar),
        count_yawns(mar)
    ]

alert, tired = [], []
for i in range(1, 6):
    for j in range(1, 3):
        filename = f'files/raw_features_{i}_{j}.json'
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                alert.extend(data['alert'])
                tired.extend(data['tired'])
        except FileNotFoundError:
            continue

X_w, y_w = [], []
for video in alert + tired:
    ear_v, mar_v, label = video['ear_values'], video['mar_values'], video['label']
    if len(ear_v) < window:
        continue
    for i in range(0, len(ear_v) - window, step):
        X_w.append(extract_features(ear_v[i:i + window], mar_v[i:i + window]))
        y_w.append(label)

X_w, y_w = np.array(X_w), np.array(y_w)
scaler_w = StandardScaler()
X_w_scaled = scaler_w.fit_transform(X_w)

model_w = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
model_w.fit(X_w_scaled, y_w)

X_v, y_v = [], []
for video in alert + tired:
    ear_v, mar_v = video['ear_values'], video['mar_values']
    if len(ear_v) < window: continue
    preds = []
    for i in range(0, len(ear_v) - window, step):
        feat = extract_features(ear_v[i:i + window], mar_v[i:i + window])
        preds.append(model_w.predict(scaler_w.transform([feat]))[0])

    if not preds:
        continue
    preds = np.array(preds)
    X_v.append([
        np.mean(preds),
        np.mean(preds[:len(preds) // 2]),
        np.mean(preds[len(preds) // 2:]),
        1 if np.sum(preds) > (len(preds) * 0.5) else 0
    ])
    y_v.append(video['label'])

X_v, y_v = np.array(X_v), np.array(y_v)
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_v, y_v, test_size=0.2, random_state=42, stratify=y_v)

scaler_v = StandardScaler()
X_train_v_scaled = scaler_v.fit_transform(X_train_v)
X_test_v_scaled = scaler_v.transform(X_test_v)

model_v = SVC(kernel='rbf', C=1.0, random_state=42)
model_v.fit(X_train_v_scaled, y_train_v)

y_pred_v = model_v.predict(X_test_v_scaled)

print(f"Точность обучения по видео: {accuracy_score(y_test_v, y_pred_v) * 100:.2f}%")

