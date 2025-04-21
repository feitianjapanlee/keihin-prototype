# 必要なライブラリのインストール
# pip install ultralytics opencv-python flask numpy

import cv2
# import base64
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response, request, jsonify
import json
import threading
import time
from collections import defaultdict

app = Flask(__name__, static_url_path='/static', static_folder='static')

# グローバル変数
model_name = "models/train61/weights/best.pt"
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

    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)  # Skip the first 500 frames to allow the camera to adjust
    
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
        
        # frame = [frame[:, :, ::-1]]  # BGR to RGB
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB

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
    # Video stream generator
    def generate():
        global lock, annotated_frame
        while True:
            if not feed_video:
                time.sleep(0.1)
                continue
            with lock:
                if annotated_frame is not None:
                    _, jpeg = cv2.imencode('.jpg', annotated_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_counts')
def get_counts():
    global counts
    return jsonify(counts)

@app.route('/toggle_stream')
def toggle_stream():
    global feed_video
    state = request.args.get('state', 'on')
    feed_video = (state == 'on')
    return jsonify({'status': 'success', 'stream_active': feed_video})

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    global detect_confi_threshold
    detect_confi_threshold = float(request.form.get('threshold', 0.5))
    return jsonify({'status': 'success', 'new_threshold': detect_confi_threshold})

@app.route('/reset_counts')
def reset_counts():
    global counts
    counts.clear()
    return jsonify({'status': 'success'})


@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>物体識別モニタリング</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .card {
                margin-bottom: 20px;
            }
            .detection-item {
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #444;
            }
            .btn-group {
                margin-bottom: 15px;
            }
            #videoContainer {
                position: relative;
            }
            #videoPlaceholder {
                display: none;
                width: 100%;
                height: 240px;
                background-color: #333;
                color: white;
                text-align: center;
                line-height: 240px;
            }
            @media (max-width: 768px) {
                .slider-container {
                    padding: 0 15px;
                }
                #videoFeed, #videoPlaceholder {
                    height: 240px;
                    line-height: 240px;
                }
            }
        </style>
    </head>
    <body class="bg-dark text-white">
        <div class="container mt-3">
            <h1 class="text-center mb-4">景品検出モニター</h1>
            
            <div class="row">
                <div class="col-12">
                    <div class="btn-group w-100">
                        <button id="toggleVideoFeedSwitch" class="btn btn-primary">ストリーム停止</button>
                        <button id="resetCounts" class="btn btn-warning">カウントリセット</button>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-8 mx-auto">
                    <div class="card bg-secondary">
                        <div class="card-body p-0" id="videoContainer">
                            <img id="videoFeed" src="/video_feed" class="img-fluid">
                            <div id="videoPlaceholder">ストリームは停止中です</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-3">
                <div class="col-md-8 mx-auto">
                    <div class="card bg-secondary">
                        <div class="card-body">
                            <h5 class="card-title">検出計数</h5>
                            <div id="countDisplay">
                                <div class="text-center">データ読み込み中...</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row mt-3">
                <div class="col-md-8 mx-auto">
                    <div class="card bg-secondary">
                        <div class="card-body">
                            <h5 class="card-title">検出閾値調整</h5>
                            <div class="slider-container">
                                <input type="range" class="form-range" min="0" max="1" step="0.1" 
                                       id="confidenceSlider" value="0.5">
                                <div class="d-flex justify-content-between">
                                    <small>0.0</small>
                                    <small>0.5</small>
                                    <small>1.0</small>
                                </div>
                                <div class="text-center mt-2">
                                    現在の閾値: <span id="thresholdValue">0.5</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // ストリームのON/OFF切り替え
            const toggleBtn = document.getElementById('toggleVideoFeedSwitch');
            const videoFeed = document.getElementById('videoFeed');
            const videoPlaceholder = document.getElementById('videoPlaceholder');
            
            toggleBtn.addEventListener('click', function() {
                if (this.textContent === 'ストリーム停止') {
                    videoFeed.style.display = 'none';
                    videoPlaceholder.style.display = 'block';
                    this.textContent = 'ストリーム開始';
                    fetch('/toggle_stream?state=off');
                } else {
                    videoFeed.style.display = 'block';
                    videoPlaceholder.style.display = 'none';
                    this.textContent = 'ストリーム停止';
                    fetch('/toggle_stream?state=on');
                }
            });
            
            // カウントリセット
            document.getElementById('resetCounts').addEventListener('click', function() {
                fetch('/reset_counts');
                // updateDetectionStats();  // 即時更新
            });
            
            // 検出閾値スライダー
            const confidenceSlider = document.getElementById('confidenceSlider');
            const thresholdValue = document.getElementById('thresholdValue');
            
            confidenceSlider.addEventListener('input', function() {
                const value = this.value;
                thresholdValue.textContent = value;
                
                // 閾値更新をサーバーに送信
                fetch('/update_threshold', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: 'threshold=' + value
                });
            });
            
        </script>
        <script>
            function updateDetectionStats() {
                fetch('/get_counts')
                    .then(response => response.json())
                    .then(data => {
                        const countDisplay = document.getElementById('countDisplay');
                        if (Object.keys(data).length === 0) {
                            countDisplay.innerHTML = '<div class="text-center">検出された物体はありません</div>';
                            return;
                        }
                        countDisplay.innerHTML = '';
                        for (const [className, count] of Object.entries(data)) {
                            const div = document.createElement('div', {class: 'detection-item'});
                            div.innerHTML = `<h3><span><img src='/static/${className}.png'>${className}:</span><span class='badge bg-primary rounded-pill'>${count}</span></h3>`;
                            countDisplay.appendChild(div);
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching counts:', error);
                    });
            };
            setInterval(updateDetectionStats, 1000);  // 1秒ごとに更新
            // 初回読み込み時にカウントを取得
            updateDetectionStats();
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    thread = threading.Thread(target=detect_objects, daemon=True)
    thread.start()
    app.run(host='0.0.0.0', port=3000)