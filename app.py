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
# MODEL_PATH = r'C:\Users\Riya Saharan\YOLOv8_Traffic_Density_Estimation\runs\detect\train2\weights\best.pt'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

model = YOLO('yolov8n.pt')
CONF_THRESHOLD = 0.50

# 1. Target Classes (COCO Dataset IDs)
TARGET_CLASSES = {
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# 2. UI-Matched Colors (BGR format for OpenCV)
CLASS_COLORS = {
    'car': (255, 229, 0),        # Cyan/Blue
    'motorcycle': (157, 255, 0),  # Green
    'bus': (53, 107, 255),       # Orange
    'truck': (255, 50, 80),      # Pink/Red
    'bicycle': (200, 0, 200)     # Purple
}


def draw_advanced_box(frame, x1, y1, x2, y2, conf, class_name, in_zone):
    color = CLASS_COLORS.get(class_name, (255, 255, 255))

    # Thicker box if the vehicle is inside our target density zone
    thickness = 3 if in_zone else 1

    label = f'{class_name.upper()} {conf:.2f}'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw centroid dot for zone tracking
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if not file or file.filename is None or file.filename == '':
        return jsonify({'error': 'No file selected'})

    ext = file.filename.rsplit('.', 1)[-1].lower()
    unique_name = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f'{unique_name}.{ext}')
    file.save(input_path)

    is_video = ext in ['mp4', 'avi', 'mov', 'mkv']

    if is_video:
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 3. Dynamic Zone Polygon (Lower-center of the camera view)
        zone_pts = np.array([
            [int(width * 0.25), int(height * 0.55)],
            [int(width * 0.75), int(height * 0.55)],
            [int(width * 0.95), int(height * 0.95)],
            [int(width * 0.05), int(height * 0.95)]
        ], np.int32).reshape((-1, 1, 2))

        output_avi = os.path.join(RESULTS_FOLDER, f'{unique_name}.avi')
        output_mp4 = os.path.join(RESULTS_FOLDER, f'{unique_name}.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # type: ignore
        out = cv2.VideoWriter(output_avi, fourcc, fps, (width, height))

        results = model.predict(
            source=input_path, stream=True, verbose=False, conf=CONF_THRESHOLD)

        all_confidences = []
        frame_vehicle_counts = []
        zone_vehicle_counts = []
        class_totals = {'car': 0, 'motorcycle': 0,
                        'bus': 0, 'truck': 0, 'bicycle': 0}

        for r in results:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Draw the Density Zone Overlay
            cv2.polylines(frame, [zone_pts], isClosed=True,
                          color=(0, 255, 255), thickness=2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone_pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            frame_count = 0
            zone_count = 0

            if r.boxes is not None:
                for box in r.boxes:
                    class_id = int(box.cls[0])

                    # 4. Filter only Target Classes
                    if class_id in TARGET_CLASSES:
                        class_name = TARGET_CLASSES[class_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        # 5. Check if vehicle is inside the target zone
                        in_zone = cv2.pointPolygonTest(
                            zone_pts, (cx, cy), False) >= 0
                        if in_zone:
                            zone_count += 1

                        frame_count += 1
                        all_confidences.append(conf)
                        class_totals[class_name] += 1

                        draw_advanced_box(frame, x1, y1, x2,
                                          y2, conf, class_name, in_zone)

            frame_vehicle_counts.append(frame_count)
            zone_vehicle_counts.append(zone_count)
            out.write(frame)

        cap.release()
        out.release()

        subprocess.run(['ffmpeg', '-y', '-loglevel', 'panic',
                       '-i', output_avi, output_mp4])
        if os.path.exists(output_avi):
            os.remove(output_avi)

        frames_processed = len(frame_vehicle_counts)
        avg_vehicles = round(sum(frame_vehicle_counts) /
                             frames_processed, 1) if frames_processed else 0
        avg_zone = round(sum(zone_vehicle_counts) /
                         frames_processed, 1) if frames_processed else 0
        avg_conf = sum(all_confidences) / \
            len(all_confidences) if all_confidences else 0

        avg_classes = {k: round(v / frames_processed, 1)
                       for k, v in class_totals.items()} if frames_processed else class_totals

        return jsonify({
            'type': 'video',
            'output': f'results/{unique_name}.mp4',
            'total_vehicles': avg_vehicles,
            'zone_density': avg_zone,
            'class_breakdown': avg_classes,
            'avg_confidence': round(avg_conf, 2),
            'frames_processed': frames_processed
        })

    else:
        # Image Processing Logic
        output_path = os.path.join(RESULTS_FOLDER, f'{unique_name}.jpg')
        img = cv2.imread(input_path)

        if img is None:
            return jsonify({'error': 'Failed to read image file'})

        height, width = img.shape[:2]
        zone_pts = np.array([
            [int(width * 0.25), int(height * 0.55)],
            [int(width * 0.75), int(height * 0.55)],
            [int(width * 0.95), int(height * 0.95)],
            [int(width * 0.05), int(height * 0.95)]
        ], np.int32).reshape((-1, 1, 2))

        cv2.polylines(img, [zone_pts], isClosed=True,
                      color=(0, 255, 255), thickness=2)
        overlay = img.copy()
        cv2.fillPoly(overlay, [zone_pts], (0, 255, 255))
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        results = model.predict(
            source=input_path, stream=False, verbose=False, conf=CONF_THRESHOLD)

        all_confidences = []
        vehicle_count = 0
        zone_count = 0
        class_totals = {'car': 0, 'motorcycle': 0,
                        'bus': 0, 'truck': 0, 'bicycle': 0}

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    if class_id in TARGET_CLASSES:
                        class_name = TARGET_CLASSES[class_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        in_zone = cv2.pointPolygonTest(
                            zone_pts, (cx, cy), False) >= 0

                        if in_zone:
                            zone_count += 1

                        vehicle_count += 1
                        class_totals[class_name] += 1
                        all_confidences.append(conf)
                        draw_advanced_box(img, x1, y1, x2, y2,
                                          conf, class_name, in_zone)

        cv2.imwrite(output_path, img)
        avg_conf = sum(all_confidences) / \
            len(all_confidences) if all_confidences else 0

        return jsonify({
            'type': 'image',
            'output': f'results/{unique_name}.jpg',
            'total_vehicles': vehicle_count,
            'zone_density': zone_count,
            'class_breakdown': class_totals,
            'avg_confidence': round(avg_conf, 2),
            'frames_processed': 1
        })


if __name__ == '__main__':
    app.run(debug=True)
