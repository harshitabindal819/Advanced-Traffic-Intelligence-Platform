from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import cv2
import uuid
import subprocess

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
MODEL_PATH = r'C:\Users\Riya Saharan\YOLOv8_Traffic_Density_Estimation\runs\detect\train2\weights\best.pt'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

# Only count detections above this threshold
CONF_THRESHOLD = 0.50


def get_box_color(conf):
    """
    Color coded by confidence level:
    🟢 GREEN  = High confidence   >= 0.80
    🟡 YELLOW = Medium confidence  0.60 - 0.79
    🔴 RED    = Low confidence     0.50 - 0.59
    """
    if conf >= 0.80:
        return (0, 220, 0)      # Green  (BGR)
    elif conf >= 0.60:
        return (0, 200, 255)    # Yellow (BGR)
    else:
        return (0, 80, 255)     # Red    (BGR)


def draw_box(frame, x1, y1, x2, y2, conf, color):
    label = f'vehicle {conf:.2f}'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    # PYLANCE FIX: Strict None/Empty check for file and filename
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
        cap.release()

        output_avi = os.path.join(RESULTS_FOLDER, f'{unique_name}.avi')
        output_mp4 = os.path.join(RESULTS_FOLDER, f'{unique_name}.mp4')

        # PYLANCE FIX: Ignore type checking for missing cv2 stub
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # type: ignore
        out = cv2.VideoWriter(output_avi, fourcc, fps, (width, height))

        cap = cv2.VideoCapture(input_path)
        results = model.predict(
            source=input_path,
            stream=True,
            verbose=False,
            conf=CONF_THRESHOLD
        )

        all_confidences = []
        frame_vehicle_counts = []

        for r in results:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # PYLANCE FIX: Handle NoneType for r.boxes length
            frame_count = len(r.boxes) if r.boxes is not None else 0
            frame_vehicle_counts.append(frame_count)

            # PYLANCE FIX: Ensure r.boxes is iterable
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    all_confidences.append(conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = get_box_color(conf)
                    draw_box(frame, x1, y1, x2, y2, conf, color)

            out.write(frame)

        cap.release()
        out.release()

        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'panic',
            '-i', output_avi, output_mp4
        ])
        if os.path.exists(output_avi):
            os.remove(output_avi)

        frames_processed = len(frame_vehicle_counts)
        avg_vehicles = round(sum(frame_vehicle_counts) /
                             frames_processed, 1) if frames_processed else 0
        avg_conf = sum(all_confidences) / \
            len(all_confidences) if all_confidences else 0

        return jsonify({
            'type': 'video',
            'output': f'results/{unique_name}.mp4',
            'total_vehicles': avg_vehicles,
            'avg_confidence': round(avg_conf, 2),
            'frames_processed': frames_processed
        })

    else:
        output_path = os.path.join(RESULTS_FOLDER, f'{unique_name}.jpg')
        img = cv2.imread(input_path)

        # PYLANCE FIX: Check if imread succeeded before predict/imwrite
        if img is None:
            return jsonify({'error': 'Failed to read image file'})

        results = model.predict(
            source=input_path,
            stream=False,
            verbose=False,
            conf=CONF_THRESHOLD
        )

        all_confidences = []
        vehicle_count = 0

        for r in results:
            # PYLANCE FIX: Handle NoneType for r.boxes length
            vehicle_count = len(r.boxes) if r.boxes is not None else 0

            # PYLANCE FIX: Ensure r.boxes is iterable
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    all_confidences.append(conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = get_box_color(conf)
                    draw_box(img, x1, y1, x2, y2, conf, color)

        cv2.imwrite(output_path, img)
        avg_conf = sum(all_confidences) / \
            len(all_confidences) if all_confidences else 0

        return jsonify({
            'type': 'image',
            'output': f'results/{unique_name}.jpg',
            'total_vehicles': vehicle_count,
            'avg_confidence': round(avg_conf, 2),
            'frames_processed': 1
        })


if __name__ == '__main__':
    app.run(debug=True)
