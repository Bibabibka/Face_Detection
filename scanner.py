import cv2
import os
import mediapipe as mp
from scipy.spatial import distance
import json

left_eye = [362, 385, 387, 263, 373, 380]
right_eye = [33, 160, 158, 133, 153, 144]


def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def calculate_mar(landmarks, w, h):
    upp = landmarks.landmark[13]
    low = landmarks.landmark[14]
    left = landmarks.landmark[78]
    right = landmarks.landmark[308]
    upper_point = (int(upp.x * w), int(upp.y * h))
    lower_point = (int(low.x * w), int(low.y * h))
    left_point = (int(left.x * w), int(left.y * h))
    right_point = (int(right.x * w), int(right.y * h))
    return distance.euclidean(upper_point, lower_point) / distance.euclidean(left_point, right_point)


def get_landmarks_coords(landmarks, ind, w, h):
    coords = []
    for i in ind:
        landmark = landmarks.landmark[i]
        coords.append((int(landmark.x * w), int(landmark.y * h)))
    return coords


def find_video(folder, name):
    for ext in ['.mov', '.mp4', '.MOV', '.MP4']:
        p = os.path.join(folder, name + ext)
        if os.path.exists(p):
            return p
    return None


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,min_tracking_confidence=0.5)

path = r"Ваш путь к файлу к папке с видео"
alert = []
tired = []

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if not os.path.isdir(folder_path): continue

    v_alert = find_video(folder_path, "0")
    v_tired = find_video(folder_path, "10")

    for video_path, label in [(v_alert, 0), (v_tired, 1)]:
        if not video_path: continue
        cap = cv2.VideoCapture(video_path)
        ear_values, mar_values = [], []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count % 3 != 0: continue

            frame = cv2.resize(frame, (640, 480))
            h, w, _ = frame.shape
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    l_eye = get_landmarks_coords(face_landmarks, left_eye, w, h)
                    r_eye = get_landmarks_coords(face_landmarks, right_eye, w, h)
                    ear_values.append(float((calculate_ear(l_eye) + calculate_ear(r_eye)) / 2.0))
                    mar_values.append(float(calculate_mar(face_landmarks, w, h)))

        cap.release()
        data = {'video_path': video_path, 'ear_values': ear_values, 'mar_values': mar_values, 'label': label}
        if label == 0:
            alert.append(data)
        else:
            tired.append(data)

with open('Сохранение метрик в ваш файл', 'w') as f:
    json.dump({'alert': alert, 'tired': tired}, f, indent=2)