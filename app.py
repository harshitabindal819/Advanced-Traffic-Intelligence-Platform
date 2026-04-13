from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import cv2
import uuid
import subprocess
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
MODEL_PATH = r'C:\Users\Riya Saharan\YOLOv8_Traffic_Density_Estimation\runs\detect\train2\weights\best.pt'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

model = YOLO('yolov8n.pt')
CONF_THRESHOLD = 0.50

TARGET_CLASSES = {
    1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 99: 'ambulance'
}

CLASS_COLORS = {
    'car': (255, 229, 0), 'motorcycle': (157, 255, 0), 'bus': (53, 107, 255),
    'truck': (255, 50, 80), 'bicycle': (200, 0, 200), 'ambulance': (0, 0, 255)
}

# A repeating color palette for dynamically generated lanes (BGR format)
LANE_COLORS = [
    (255, 100, 0),   # Blue
    (0, 255, 255),   # Yellow
    (0, 100, 255),   # Orange
    (255, 0, 255),   # Purple
    (0, 255, 0),     # Green
    (255, 255, 0)    # Cyan
]


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
    lanes = data.get('lanes')  # Variable length list of 4-point arrays
    simulate_emergency = data.get('simulate_emergency', False)

    if not video_id or not lanes or len(lanes) == 0:
        return jsonify({'error': 'Invalid calibration data. At least one lane required.'})

    input_path = os.path.join(UPLOAD_FOLDER, video_id)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Dynamically build lane polygons
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

    for r in results:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frames_processed += 1
        current_frame_lanes = {f'Lane {i+1}': 0 for i in range(len(lanes))}

        # Draw dynamic overlays
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
                    # Check which dynamic lane the vehicle is in
                    for i, poly in enumerate(lane_polygons):
                        if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                            current_frame_lanes[f'Lane {i+1}'] += 1
                            total_lane_counts[f'Lane {i+1}'] += 1
                            in_zone = True
                            break  # Only count in one lane

                    all_confidences.append(conf)
                    draw_advanced_box(frame, x1, y1, x2, y2,
                                      conf, class_name, in_zone)

        # Draw dynamic text above each lane
        for i, poly in enumerate(lane_polygons):
            l_x = int(np.mean(poly[:, 0, 0]))
            text_y = int(np.min(poly[:, 0, 1])) - 15
            color = LANE_COLORS[i % len(LANE_COLORS)]
            cv2.putText(frame, f"L{i+1}: {current_frame_lanes[f'Lane {i+1}']}",
                        (l_x-30, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)

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

    optimal_lane = min(avg_lanes.keys(), key=lambda k: avg_lanes[k])

    return jsonify({
        'type': 'video',
        'output': f'results/{video_id}_final.mp4',
        'lane_densities': avg_lanes,
        'recommended_lane': optimal_lane,
        'emergency_detected': emergency_detected,
        'avg_confidence': round(avg_conf, 2),
        'frames_processed': frames_processed
    })


if __name__ == '__main__':
    app.run(debug=True)
