from maix import camera, image, nn, app, tracker
from flask import Flask, Response
import cv2  # only for JPEG encode
import json
import threading
import time
from collections import defaultdict

fapp = Flask(__name__)

# グローバル変数
counts = defaultdict(int)
last_counts = defaultdict(int)
track_history = defaultdict(list)
detected_objects = []   # list of dict for display result
lock = threading.Lock()
annotated_frame = None

detect_confi_threshold = 0.5        # confidence threshold of detection
detect_iou_threshold = 0.45         # iou threshold of detection
max_lost_buff_frame = 120           # frames to keep before mark as lost
track_threshold = 0.4               # confidence threshold to continue track
high_threshold = 0.6                # confidence threshold to new a track
match_threshold = 0.8               # iou threshold to treat as same object
max_history_num = 5                 # max length of track history


def detect_objects():
    global counts, track_history, detected_objects, lock, annotated_frame
    
    # detector = nn.YOLOv8(model="/root/models/yolov8n.mud", dual_buff = True)
    detector = nn.YOLOv8(model="/root/models/train15_yolov8n_int8.mud", dual_buff = True)
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
                counts[detector.labels[obj.class_id]] += 1
                counted_ids.add(track.id)
        
        # remove some history do not need
        if len(counted_ids) > 200:
            counted_ids = counted_ids[100:]

        # draw tracking objects with red color
        with lock:
            for obj in detected_objects:
                annotated_frame.draw_rect(obj['x'], obj['y'], obj['w'], obj['h'], color = image.COLOR_RED)
                msg = f"{obj['class']}({obj['id']}): {obj['score']:.2f}"
                annotated_frame.draw_string(obj['x'], obj['y'], msg, color = image.COLOR_RED)

        time.sleep(0.1)

@fapp.route('/video_feed')
def video_feed():
    def generate():
        global lock, annotated_frame
        while True:
            with lock:
                if annotated_frame is not None:
                    _, jpeg = cv2.imencode('.jpg', image.image2cv(annotated_frame))
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.05)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@fapp.route('/stream')
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

            # time.sleep(0.5)
   
    return Response(event_stream(), mimetype="text/event-stream")

@fapp.route('/')
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
    fapp.run(host='0.0.0.0', port=5000)
