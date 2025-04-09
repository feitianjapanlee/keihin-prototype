# 必要なライブラリのインストール
# pip install ultralytics opencv-python flask numpy

import cv2
from ultralytics import YOLO
from flask import Flask, Response
import json
import threading
import time
from collections import defaultdict

app = Flask(__name__)

# グローバル変数
model_name = "models/train44/weights/best.onnx"
counts = defaultdict(int)
last_counts = defaultdict(int)
track_history = defaultdict(list)
detected_objects = []   # list of dict for display result
lock = threading.Lock()
annotated_frame = None
draw_rects = False                  # whether to draw image of detected objects
feed_video = False                  # whether to feed result image to web

detect_confi_threshold = 0.7        # confidence threshold of detection
detect_iou_threshold = 0.5          # iou threshold of detection
max_lost_buff_frame = 30            # frames to keep before mark as lost
track_threshold = 0.4               # confidence threshold to continue track
high_threshold = 0.6                # confidence threshold to new a track
match_threshold = 0.8               # iou threshold to treat as same object
max_history_num = 5                 # max length of track history

cap = None

def init_camera():
    global cap
    # MaixCam用設定 (適宜調整)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("Error: カメラを開けません")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return True

def detect_objects():
    global counts, track_history, detected_objects, lock, annotated_frame, model_name, draw_rects
    
    if not init_camera():
        return

    # model = YOLO('models/yolov8n.pt')
    model = YOLO(model_name)
    
    counted_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("カメラからフレームを取得できません")
            time.sleep(1)
            continue
        
        # 推論 (軽量化のため解像度を下げる)
        results = model.track(frame, imgsz=320, persist=True)
        
        if draw_rects:
            with lock:
                annotated_frame = results[0].plot()

        current_objects = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            for box, track_id, cls in zip(boxes, track_ids, clss):
                x, y, w, h = box
                current_objects.append({
                    'id': track_id,
                    'class': model.names[int(cls)],
                    'x': float(x),
                    'y': float(y),
                    'w': float(w),
                    'h': float(h)
                })
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30: # 30フレーム保持
                    track.pop(0)
                if len(track) >= 5: # 5フレーム以上安定した
                    if track_id not in counted_ids:
                        counts[model.names[cls]] += 1
                        counted_ids.add(track_id)

        detected_objects = current_objects
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    global feed_video
    def generate():
        global lock, annotated_frame
        while True:
            with lock:
                if annotated_frame is not None:
                    _, jpeg = cv2.imencode('.jpg', annotated_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.05)
    if feed_video:
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(status=204)

@app.route('/stream')
def stream():
    def event_stream():
        global counts, last_counts, detected_objects
        while True:
            # カウントに変更があった場合のみ送信
            if counts != last_counts:
                data = {
                    'counts': dict(counts),
                    'objects': detected_objects
                }
                json_data = json.dumps(data)
                yield f"data: {json_data}\n\n"
                last_counts = counts.copy()

            time.sleep(0.5)
   
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/')
def index():
    return """
    <html>
    <head>
        <meta charset="UTF-8">
        <style type="text/css">
            #count-display { font-size: 1.6em }
            .count-value { font-size: 1.6em; color: red; }
        </style>
    </head>
    <body>
        <h1>景品検出モニター</h1>

        <div class="container">
            <div>
                <img id="video-feed" src="/video_feed" width="640">
            </div>
            
            <div class="count-panel">
                <h2>カウント結果</h2>
                <div id="count-display">
                    <!-- カウントデータがここに表示されます -->
                </div>
            </div>
            
        </div>

        <script>
            const evtSource = new EventSource("/stream");
            evtSource.onmessage = function(event) {
                const newCounts = JSON.parse(event.data);
                console.info("更新:", newCounts);
                // カウント表示を更新
                const countDisplay = document.getElementById('count-display');
                countDisplay.innerHTML = '';
                for (const [className, count] of Object.entries(newCounts.counts)) {
                    const div = document.createElement('div');
                    div.innerHTML = `<strong>${className}:</strong> <span class="count-value">${count}</span>`;
                    countDisplay.appendChild(div);
                }
            };
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    thread = threading.Thread(target=detect_objects, daemon=True)
    thread.start()
    app.run(host='0.0.0.0', port=5000)