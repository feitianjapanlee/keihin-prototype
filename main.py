from maix import camera, image, nn, app, tracker, touchscreen, display, sys
from flask import Flask, Response
import cv2  # only for JPEG encode
import base64
import json
import threading
import time
from collections import defaultdict

fapp = Flask(__name__)

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

detect_confi_threshold = 0.7        # confidence threshold of detection
detect_iou_threshold = 0.3          # iou threshold of detection
max_lost_buff_frame = 20            # frames to keep before mark as lost
track_threshold = 0.4               # confidence threshold to continue track
high_threshold = 0.7                # confidence threshold to new a track
match_threshold = 0.5               # iou threshold to treat as same object
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
                counts[detector.labels[obj.class_id]] += 1
                counted_ids.add(track.id)
        
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
    if not feed_video:
        img = image.Image(640, 480)
        img.draw_string(20, 200, "MONITOR OFF", image.COLOR_GRAY)
        _, jpeg = cv2.imencode('.jpg', image.image2cv(img))
        return Response(b'data:image/jpeg;base64,' + base64.b64encode(jpeg))

    def generate():
        global lock, annotated_frame
        while True:
            if annotated_frame is not None:
                with lock:
                    _, jpeg = cv2.imencode('.jpg', image.image2cv(annotated_frame))
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.2)
    
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