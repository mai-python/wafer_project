from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import serial
from ultralytics import YOLO

app = Flask(__name__)

# 시스템 상태
status = {
    "current": "standby",
    "recognition": "waiting",
    "motor": "standby",
    "error": "none",
    "wafer_type": "none",
    "dx": 0.0,
    "dy": 0.0,
    "time": 0.0
}

log_data = ["[00:00] Log initialized"]
command = "stop"

# 설정값
HOME_OFFSET = (0, 0)
STEPS_PER_MM = 1600

# YOLO 모델 로드 및 카메라 연결
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

try:
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
except:
    ser = None
    log_data.append("[WARN] Arduino not connected.")

@app.route('/')
@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify(status)

@app.route('/logs')
def get_logs():
    return jsonify(logs=log_data[-50:])

@app.route('/send_command', methods=['POST'])
def send_command():
    global command
    action = request.json.get("action")
    if action == "start":
        command = "start"
        status["current"] = "operating"
        status["motor"] = "standby"
        log_data.append("[Server] Command received: start")
    elif action == "pause":
        command = "stop"
        status["current"] = "stopped"
        status["motor"] = "stopped"
        log_data.append("[Server] Command received: pause")
    elif action == "reset":
        command = "stop"
        status.update({
            "current": "standby",
            "recognition": "waiting",
            "motor": "standby",
            "error": "none",
            "wafer_type": "none",
            "dx": 0.0,
            "dy": 0.0,
            "time": 0.0
        })
        log_data.append("[Server] System reset to standby")
    elif action == "home":
        dx_home = HOME_OFFSET[0]
        dy_home = HOME_OFFSET[1]
        step_x = int(abs(dx_home) * STEPS_PER_MM)
        step_y = int(abs(dy_home) * STEPS_PER_MM)
        dir_x = 1 if dx_home >= 0 else 0
        dir_y = 1 if dy_home >= 0 else 0
        send_to_arduino(f"{dir_x},{dir_y},{step_x},{step_y}")
        status["motor"] = "moving"
        log_data.append("[Server] Home command → Return to origin")
    return jsonify({"status": "ok"})

def send_to_arduino(cmd_str):
    if ser and ser.is_open:
        try:
            ser.write((cmd_str + "\n").encode())
            log_data.append(f"[Serial] Sent to Arduino: {cmd_str}")
        except Exception as e:
            status["error"] = "Serial Error"
            log_data.append(f"[Serial Error] {str(e)}")
    else:
        log_data.append("[Serial] Port not open")

def gen_frames():
    global command
    while True:
        success, frame = cap.read()
        if not success:
            continue

        if command == "start":
            start_time = time.time()
            results = model(frame)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                dx = cx - (frame.shape[1] // 2)
                dy = cy - (frame.shape[0] // 2)

                status["dx"] = round(dx * 0.026, 2)
                status["dy"] = round(dy * 0.026, 2)
                status["recognition"] = "done"
                status["wafer_type"] = "P-100"
                status["time"] = round(time.time() - start_time, 2)

                step_x = int(abs(dx) * 0.026 * STEPS_PER_MM)
                step_y = int(abs(dy) * 0.026 * STEPS_PER_MM)
                dir_x = 1 if dx >= 0 else 0
                dir_y = 1 if dy >= 0 else 0
                send_to_arduino(f"{dir_x},{dir_y},{step_x},{step_y}")

                status["motor"] = "moving"
            else:
                status["recognition"] = "none"
                status["motor"] = "standby"

            command = "stop"

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

