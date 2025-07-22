import cv2
import numpy as np
from ultralytics import YOLO

# Симуляция камер
ENABLED_CAMERAS = [True, True, True]  # [B, G, R]
def simulate_simple_rgb_split(frame):
    frame = frame.astype(np.float32)
    result = np.zeros_like(frame)
    for i, enabled in enumerate(ENABLED_CAMERAS):
        if enabled:
            result[:, :, i] = frame[:, :, i]
    return np.clip(result, 0, 255).astype(np.uint8)

# Коррекция
def increase_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def increase_saturation(image, factor=1.3):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def enhance_image(image):
    image = increase_contrast(image)
    image = increase_saturation(image)
    return image

# Функция детекции
def process_video_with_yolo(model_path, input_video_path, output_video_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Ошибка открытия видео.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    CONFIDENCE_THRESHOLD = 0.3

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        simulated = simulate_simple_rgb_split(frame)

        corrected = enhance_image(simulated)

        results = model.track(corrected, conf=CONFIDENCE_THRESHOLD, persist=True)[0]

        annotated = corrected.copy()
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, 'Fish', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out.write(annotated)

    cap.release()
    out.release()
    print(f"Видео сохранено: {output_video_path}")
