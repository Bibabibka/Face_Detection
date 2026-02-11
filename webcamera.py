import pickle
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
import time

with open('SVC_model.pkl', 'rb') as f:
    models = pickle.load(f)

model_w, scaler_w = models['model_w'], models['scaler_w']
model_v, scaler_v = models['model_v'], models['scaler_v']

left_eye = [362, 385, 387, 263, 373, 380]
right_eye = [33, 160, 158, 133, 153, 144]
window = 180
step = 60


def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def calculate_mar(landmarks, w, h):
    upp = landmarks.landmark[13]
    low = landmarks.landmark[14]
    left = landmarks.landmark[78]
    right = landmarks.landmark[308]
    u_p = (int(upp.x * w), int(upp.y * h))
    l_p = (int(low.x * w), int(low.y * h))
    lt_p = (int(left.x * w), int(left.y * h))
    rt_p = (int(right.x * w), int(right.y * h))
    return distance.euclidean(u_p, l_p) / distance.euclidean(lt_p, rt_p)


def count_yawns(mar_values):
    y_c, prog, f_c = 0, False, 0
    for mar in mar_values:
        if mar > 0.6:
            if not prog:
                prog, f_c = True, 1
            else:
                f_c += 1
        else:
            if prog and f_c >= 45: y_c += 1
            prog, f_c = False, 0
    return y_c


def extract_features(ear, mar):
    per = sum(1 for e in ear if e < 0.2) / len(ear)
    return [np.mean(ear), np.min(ear), np.std(ear), per, np.mean(mar), np.max(mar), count_yawns(mar)]


def get_landmarks_coords(landmarks, ind, w, h):
    return [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in ind]


yawn_counter = 0
yawn_in_progress = False
yawn_start_time = None

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,min_tracking_confidence=0.5)
ear_values, mar_values = [], []

s = 0
cap = cv2.VideoCapture(s)
while cv2.waitKey(1) != ord('q'):
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=frame,landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_TESSELATION,landmark_drawing_spec=None,connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            left_eye_coords = get_landmarks_coords(face_landmarks, left_eye, w, h)
            right_eye_coords = get_landmarks_coords(face_landmarks, right_eye, w, h)
            left_ear = calculate_ear(left_eye_coords)
            right_ear = calculate_ear(right_eye_coords)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(face_landmarks, w, h)

            ear_values.append(float(avg_ear))
            mar_values.append(float(mar))

            cv2.putText(frame, "Ear: {:.2f}".format(avg_ear), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Mar: {:.2f}".format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Yawn_counter: {:.2f}".format(yawn_counter), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)

            if mar > 0.6:
                if not yawn_in_progress:
                    yawn_start_time = time.time()
                    yawn_in_progress = True
            else:
                if yawn_in_progress:
                    yawn_duration = time.time() - yawn_start_time
                    if yawn_duration >= 1.5:
                        yawn_counter += 1
                    yawn_in_progress = False

    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
face_mesh.close()

if len(ear_values) >= window:
    preds = []
    for i in range(0, len(ear_values) - window, step):
        feat = extract_features(ear_values[i:i + window], mar_values[i:i + window])
        preds.append(model_w.predict(scaler_w.transform([feat]))[0])

    preds = np.array(preds)
    v_feat = [
        np.mean(preds),
        np.mean(preds[:len(preds) // 2]),
        np.mean(preds[len(preds) // 2:]),
        1 if np.sum(preds) > (len(preds) * 0.5) else 0
    ]
    final = model_v.predict(scaler_v.transform([v_feat]))[0]
    status = "Уставший" if final == 1 else "Бодрый"
    print(f"Итог: {status}")
else:
    print("Короткое видео")