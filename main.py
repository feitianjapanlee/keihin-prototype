from maix import camera, image, nn, app, tracker, touchscreen, display, sys
from flask import Flask, Response, request, jsonify
import cv2  # only for JPEG encode
import json
import threading
import time
from collections import defaultdict
# import numpy as np

fapp = Flask(__name__, static_url_path='/static', static_folder='/root/static')

# グローバル変数
model_name = "/root/models/train48_yolo11n_320.mud"
counts = defaultdict(int)
last_counts = defaultdict(int)
track_history = defaultdict(list)
detected_objects = []   # list of dict for display result
lock = threading.Lock()
annotated_frame = None
draw_detect = True     # whether to draw image of detected objects
feed_video = True      # whether to feed result image to web

detect_confi_threshold = 0.5        # confidence threshold of detection
detect_iou_threshold = 0.5          # iou threshold of detection
max_lost_buff_frame = 10            # frames to keep before mark as lost
track_threshold = 0.4               # confidence threshold to continue track
high_threshold = 0.5                # confidence threshold to new a track
match_threshold = 0.8               # iou threshold to treat as same object
max_history_num = 5                 # max length of track history


def detect_objects():
    global model_name, counts, track_history, detected_objects, lock, annotated_frame
    
    # detector = nn.YOLOv8(model="/root/models/yolov8n.mud", dual_buff = True)
    detector = nn.YOLOv8(model=model_name, dual_buff = True)
    cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())    
    maix_tracker = tracker.ByteTracker(max_lost_buff_frame, track_threshold, high_threshold, match_threshold, max_history_num)
    
    counted_ids = set()

    while not app.need_exit():

        frame = cam.read()

        # 推論 (軽量化のため解像度を下げる)
        # results = model.track(frame, imgsz=320, persist=True)
        results = detector.detect(frame, conf_th = detect_confi_threshold, iou_th = detect_iou_threshold)
        
        # draw detected results before tracking with gray color
        with lock:
            annotated_frame = frame.copy()
            if draw_detect:
                for obj in results:
                    annotated_frame.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_GRAY)
                    msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
                    annotated_frame.draw_string(obj.x, obj.y, msg, color = image.COLOR_GRAY)

        maix_objects = []   # list used as input parameter for tracker function
        for obj in results:
            maix_objects.append(tracker.Object(obj.x, obj.y, obj.w, obj.h, obj.class_id, obj.score))

        tracking_objects = []   # list of Track class
        tracking_objects = maix_tracker.update(maix_objects)

        '''
        if len(tracking_objects) > 0:
            for t in tracking_objects:
                print(f'id={t.id} score={t.score:.2f} lost={t.lost} start={t.start_frame_id} frame={t.frame_id} hislen={len(t.history)}¥n')
        '''

        detected_objects = []   # JSON serializable dict list
        for track in tracking_objects:
            if track.lost:
                continue
            obj = track.history[-1]
            detected_objects.append({
                'id': track.id,
                'class': detector.labels[obj.class_id],
                'x': obj.x,
                'y': obj.y,
                'w': obj.w,
                'h': obj.h,
                'score': track.score
            })
            if track.id not in counted_ids:
                print(f"track.id: {track.id}, cls: {detector.labels[obj.class_id]} is new, add to counted_ids.")  # デバッグメッセージ
                counts[detector.labels[obj.class_id]] += 1
                counted_ids.add(track.id)
            else:
                print(f"track.id: {track.id}, cls: {detector.labels[obj.class_id]} is already counted.")  # デバッグメッセージ

        # remove some history do not need
        if len(counted_ids) > 200:
            counted_ids = counted_ids[100:]

        # draw tracking objects with red color
        if draw_detect:
            with lock:
                for obj in detected_objects:
                    annotated_frame.draw_rect(obj['x'], obj['y'], obj['w'], obj['h'], color = image.COLOR_RED)
                    msg = f"{obj['class']}({obj['id']}): {obj['score']:.2f}"
                    annotated_frame.draw_string(obj['x'], obj['y'], msg, color = image.COLOR_RED)

        time.sleep(0.1)

@fapp.route('/video_feed')
def video_feed():
    def generate():
        global lock, annotated_frame, feed_video
        while True:
            if not feed_video:
                time.sleep(0.1)
                continue
            with lock:
                if annotated_frame is not None:
                    _, jpeg = cv2.imencode('.jpg', image.image2cv(annotated_frame))
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@fapp.route('/counts_updated')
def counts_updated():
    def event_stream():
        global counts, last_counts, detected_objects
        while True:
            # カウントに変更があった場合のみ送信
            if counts != last_counts:
                data = {
                    'counts': dict(counts),
                    # 'objects': detected_objects
                }
                json_data = json.dumps(data)
                yield f"data: {json_data}\n\n"
                last_counts = counts.copy()

            time.sleep(0.1)
   
    return Response(event_stream(), mimetype="text/event-stream")

@fapp.route('/get_counts')
def get_counts():
    global counts
    return jsonify(counts)

@fapp.route('/toggle_video')
def toggle_video():
    global feed_video
    state = request.args.get('state', 'on')
    feed_video = (state == 'on')
    return jsonify({'status': 'success', 'video_active': feed_video})

@fapp.route('/update_threshold', methods=['POST'])
def update_threshold():
    global detect_confi_threshold
    detect_confi_threshold = float(request.form.get('threshold', 0.5))
    return jsonify({'status': 'success', 'new_threshold': detect_confi_threshold})

@fapp.route('/reset_counts')
def reset_counts():
    global counts
    counts.clear()
    return jsonify({'status': 'success'})


@fapp.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>景品検出モニター</title>
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
                    fetch('/toggle_video?state=off');
                } else {
                    videoFeed.style.display = 'block';
                    videoPlaceholder.style.display = 'none';
                    this.textContent = 'ストリーム停止';
                    fetch('/toggle_video?state=on');
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
        <script>
            const evtSource = new EventSource("/counts_updated");
            evtSource.onmessage = function(event) {
                const newCounts = JSON.parse(event.data);
                console.info("更新:", newCounts);
                // カウント表示を更新
                updateDetectionStats();
            };
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    disp = display.Display()
    img = image.Image(disp.width(), disp.height())
    
    img.draw_string(0, 10, "Keihin Counter Prototype", image.COLOR_WHITE, 2)
    img.draw_string(0, 50, f"{model_name}", image.COLOR_WHITE, 1)
    img.draw_string(0, 80, f"http://{sys.ip_address()['wlan0']}:5000", image.COLOR_GREEN, 2)
    disp.show(img)

    thread = threading.Thread(target=detect_objects, daemon=True)
    thread.start()
    fapp.run(host='0.0.0.0', port=5000)

    """
    ts = touchscreen.TouchScreen()
    exit_label = "[Exit]"
    exit_ssize = image.string_size(exit_label)
    exit_btn_pos = [disp.width()-exit_ssize.width()-9*2, disp.height()-exit_ssize.height()-9*2, exit_ssize.width()+8*2, exit_ssize.height()+8*2]
    img.draw_string(exit_btn_pos[0]+8, exit_btn_pos[1]+8, exit_label, image.COLOR_WHITE)
    img.draw_rect(exit_btn_pos[0], exit_btn_pos[1], exit_btn_pos[2], exit_btn_pos[3], image.COLOR_GREEN)
    def button_event(x, y, btn_pos):
        return x>btn_pos[0] and x<btn_pos[0]+btn_pos[2] and y>btn_pos[1] and y<btn_pos[1]+btn_pos[3]
    while not app.need_exit():
        x, y, pressed = ts.read()
        if button_event(x, y, exit_btn_pos):
            app.set_exit_flag(True)
    """