from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import cv2
import uuid
import subprocess
import numpy as np
import easyocr

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize Models
print("Loading YOLOv8 and EasyOCR... Please wait.")
model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['en'], gpu=False)
CONF_THRESHOLD = 0.50

TARGET_CLASSES = {
    1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 99: 'ambulance'
}

CLASS_COLORS = {
    'car': (255, 229, 0), 'motorcycle': (157, 255, 0), 'bus': (53, 107, 255),
    'truck': (255, 50, 80), 'bicycle': (200, 0, 200), 'ambulance': (0, 0, 255)
}

LANE_COLORS = [
    (255, 100, 0), (0, 255, 255), (0, 100,
                                   255), (255, 0, 255), (0, 255, 0), (255, 255, 0)
]

# --- NEW STRICT PLATE VALIDATION FUNCTION ---


def is_valid_plate(text):
    # Must be standard plate length
    if not (4 <= len(text) <= 10):
        return False

    # Count letters and numbers
    letters = sum(c.isalpha() for c in text)
    numbers = sum(c.isdigit() for c in text)

    # Real plates usually have at least 1 letter and at least 2 numbers
    if letters >= 1 and numbers >= 2:
        return True

    return False


def draw_advanced_box(frame, x1, y1, x2, y2, conf, class_name, in_zone):
    color = CLASS_COLORS.get(class_name, (255, 255, 255))
    thickness = 3 if class_name == 'ambulance' else (2 if in_zone else 1)
    label = f'{class_name.upper()} {conf:.2f}'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if not file or file.filename is None or file.filename == '':
        return jsonify({'error': 'No file selected'})

    ext = file.filename.rsplit('.', 1)[-1].lower()
    unique_name = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f'{unique_name}.{ext}')
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': 'Could not read video file.'})

    frame_name = f'{unique_name}_frame.jpg'
    frame_path = os.path.join(RESULTS_FOLDER, frame_name)
    cv2.imwrite(frame_path, frame)

    return jsonify({'video_id': f'{unique_name}.{ext}', 'frame_url': f'results/{frame_name}'})


@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if data is None:
        return jsonify({'error': 'No data provided'})

    video_id = data.get('video_id')
    lanes = data.get('lanes')
    simulate_emergency = data.get('simulate_emergency', False)

    if not video_id or not lanes or len(lanes) == 0:
        return jsonify({'error': 'Invalid calibration data. At least one lane required.'})

    input_path = os.path.join(UPLOAD_FOLDER, video_id)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    lane_polygons = [np.array(pts, np.int32).reshape((-1, 1, 2))
                     for pts in lanes]

    output_avi = os.path.join(RESULTS_FOLDER, f'{video_id}.avi')
    output_mp4 = os.path.join(RESULTS_FOLDER, f'{video_id}_final.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # type: ignore
    out = cv2.VideoWriter(output_avi, fourcc, fps, (width, height))

    results = model.predict(source=input_path, stream=True,
                            verbose=False, conf=CONF_THRESHOLD)

    all_confidences = []
    frames_processed = 0
    total_lane_counts = {f'Lane {i+1}': 0 for i in range(len(lanes))}
    emergency_detected = simulate_emergency
    detected_plates = set()

    for r in results:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frames_processed += 1
        current_frame_lanes = {f'Lane {i+1}': 0 for i in range(len(lanes))}

        overlay = frame.copy()
        for i, poly in enumerate(lane_polygons):
            color = LANE_COLORS[i % len(LANE_COLORS)]
            cv2.fillPoly(overlay, [poly], color)

        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, lane_polygons, isClosed=True,
                      color=(255, 255, 255), thickness=1)

        if r.boxes is not None:
            for box in r.boxes:
                class_id = int(box.cls[0])
                if class_id in TARGET_CLASSES:
                    class_name = TARGET_CLASSES[class_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if class_name == 'ambulance':
                        emergency_detected = True

                    in_zone = False
                    for i, poly in enumerate(lane_polygons):
                        if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                            current_frame_lanes[f'Lane {i+1}'] += 1
                            total_lane_counts[f'Lane {i+1}'] += 1
                            in_zone = True
                            break

                    # --- OPTIMIZED ANPR PIPELINE ---
                    if frames_processed % 10 == 0 and (x2 - x1) > 60 and (y2 - y1) > 60:
                        crop_y1 = y1 + int((y2 - y1) * 0.5)
                        vehicle_crop = frame[crop_y1:y2, x1:x2]

                        if vehicle_crop.size > 0:
                            try:
                                ocr_results = reader.readtext(vehicle_crop)
                                for res in ocr_results:
                                    if len(res) == 3:
                                        _, text, prob = res
                                        if float(prob) > 0.4:
                                            clean_text = "".join(
                                                e for e in text if e.isalnum()).upper()

                                            # Call the new central validation function
                                            if is_valid_plate(clean_text):
                                                detected_plates.add(clean_text)
                            except Exception as e:
                                print(f"OCR Error: {e}")

                    all_confidences.append(conf)
                    draw_advanced_box(frame, x1, y1, x2, y2,
                                      conf, class_name, in_zone)

        for i, poly in enumerate(lane_polygons):
            l_x = int(np.mean(poly[:, 0, 0]))
            text_y = int(np.min(poly[:, 0, 1])) - 15
            color = LANE_COLORS[i % len(LANE_COLORS)]
            cv2.putText(frame, f"L{i+1}: {current_frame_lanes[f'Lane {i+1}']}",
                        (l_x-30, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)

    # --- DIRECT PLATE FALLBACK (For testing raw plate images) ---
    if frames_processed == 1 and len(detected_plates) == 0:
        cap = cv2.VideoCapture(input_path)
        ret, frame = cap.read()
        if ret and frame is not None:
            try:
                ocr_results = reader.readtext(frame)
                for res in ocr_results:
                    if len(res) == 3:
                        _, text, prob = res
                        if float(prob) > 0.3:
                            clean_text = "".join(
                                e for e in text if e.isalnum()).upper()

                            # Call the new central validation function here too!
                            if is_valid_plate(clean_text):
                                detected_plates.add(clean_text)
            except Exception as e:
                print(f"Fallback OCR Error: {e}")

    cap.release()
    out.release()

    subprocess.run(['ffmpeg', '-y', '-loglevel', 'panic',
                   '-i', output_avi, output_mp4])
    if os.path.exists(output_avi):
        os.remove(output_avi)

    avg_conf = sum(all_confidences) / \
        len(all_confidences) if all_confidences else 0

    if frames_processed > 0:
        avg_lanes = {k: round(v / frames_processed, 1)
                     for k, v in total_lane_counts.items()}
    else:
        avg_lanes = {k: float(v) for k, v in total_lane_counts.items()}

    optimal_lane = min(
        avg_lanes.keys(), key=lambda k: avg_lanes[k]) if avg_lanes else "N/A"

    return jsonify({
        'type': 'video',
        'output': f'results/{video_id}_final.mp4',
        'lane_densities': avg_lanes,
        'recommended_lane': optimal_lane,
        'emergency_detected': emergency_detected,
        'plates': list(detected_plates)[:15],
        'avg_confidence': round(avg_conf, 2),
        'frames_processed': frames_processed
    })


if __name__ == '__main__':
    app.run(debug=True)
