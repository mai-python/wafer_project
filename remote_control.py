
from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading

app = Flask(__name__)

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
    return jsonify(logs=log_data)

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
    return jsonify({"status": "ok"})

def gen_frames():
    while True:
        frame = cv2.imread("static/no_signal.png")
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

