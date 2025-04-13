# 必要なライブラリのインストール
# pip install ultralytics opencv-python flask numpy

import cv2
# import base64
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response
import json
import threading
import time
from collections import defaultdict

app = Flask(__name__)

# グローバル変数
model_name = "models/train48/weights/best.onnx"
counts = defaultdict(int)
last_counts = defaultdict(int)
track_history = defaultdict(list)
detected_objects = []   # list of dict for display result
lock = threading.Lock()
annotated_frame = None
draw_detect = True                  # whether to draw image of detected objects
feed_video = True                  # whether to feed result image to web

detect_confi_threshold = 0.5        # confidence threshold of detection
detect_iou_threshold = 0.5          # iou threshold of detection
# max_lost_buff_frame = 30            # frames to keep before mark as lost
# track_threshold = 0.4               # confidence threshold to continue track
# high_threshold = 0.7                # confidence threshold to new a track
# match_threshold = 0.5               # iou threshold to treat as same object
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
    global counts, track_history, detected_objects, lock, annotated_frame, model_name, draw_detect, cap
    global feed_video, detect_confi_threshold, detect_iou_threshold, max_history_num, count_up_history_num
    global last_counts
    
    if not init_camera():
        return

    # model = YOLO('models/yolov8n.pt')
    model = YOLO(model_name, task='detect')
    
    counted_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("カメラからフレームを取得できません")
            time.sleep(1)
            continue
        
        # print("フレームを取得しました")  # デバッグメッセージ

        # 推論 (軽量化のため解像度を下げる)
        results = model.track(
            frame, 
            imgsz=320, 
            conf=detect_confi_threshold, 
            iou=detect_iou_threshold, 
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False # 推論結果の詳細を表示
        )
        
        if draw_detect:
            with lock:
                annotated_frame = results[0].plot()

        current_objects = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()

            # print(f"track_ids len: {len(track_ids)}")  # デバッグメッセージ

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
                # print(f"track_id: {track_id}, cls: {model.names[int(cls)]}, x: {x}, y: {y}")  # デバッグメッセージ
                track = track_history[track_id]
                # print(f"get track history of {track_id} len: {len(track)}")  # デバッグメッセージ
                track.append((float(x), float(y)))
                # print(f"append to history of track_id: {track_id}")  # デバッグメッセージ
                if len(track) > max_history_num:
                    track.pop(0)
                    # print(f"pop 0 of track_id: {track_id},{model.names[int(cls)]} for len > {max_history_num}")  # デバッグメッセージ
                # if len(track) >= count_up_history_num:
                    # print(f"track_id: {track_id}, cls: {model.names[int(cls)]}, len > {count_up_history_num}")  # デバッグメッセージ
                if track_id not in counted_ids:
                    print(f"track_id: {track_id}, cls: {model.names[int(cls)]} is new, add to counted_ids.")  # デバッグメッセージ
                    counts[model.names[cls]] += 1
                    counted_ids.add(track_id)
                else:
                    print(f"track_id: {track_id}, cls: {model.names[int(cls)]} is already counted.")  # デバッグメッセージ

        # remove some history do not need
        if len(counted_ids) > 200:
            counted_ids = counted_ids[100:]

        # print(f"counts: {counts}")  # デバッグメッセージ
        detected_objects = current_objects
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    global feed_video
    if not feed_video:
        # Create a black image with "MONITOR OFF" text
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "MONITOR OFF", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, jpeg = cv2.imencode('.jpg', img)
        return Response(b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n',
                       mimetype='multipart/x-mixed-replace; boundary=frame')

    # Video stream generator
    def generate():
        global lock, annotated_frame
        while True:
            with lock:
                if annotated_frame is not None:
                    _, jpeg = cv2.imencode('.jpg', annotated_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

            time.sleep(0.1)
   
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
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
    app.run(host='0.0.0.0', port=3000)