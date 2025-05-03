import sys
import cv2
import numpy as np
import asyncio
import serial_asyncio
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
from qasync import QEventLoop
import os

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
Target_Point = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]
TOLERANCE_PX = 1
PIXEL_TO_MM_X = 0.6
PIXEL_TO_MM_Y = 0.83
STEPS_PER_MM = 55
MAX_STEPS = 32000
BAUDRATE = 9600
SERIAL_PORT = "/dev/ttyACM1"
CAMERA_INDEX = 0
HOME_DXDY = (500, 300)

YELLOW_LOWER = (15, 100, 100)
YELLOW_UPPER = (40, 255, 255)
AREA_THRESHOLD = 1000
ROUNDNESS_THRESHOLD = 0.4

model = YOLO("yolov8n.pt")

confirmed_center = None
send_enabled = False
sending_command = False
awaiting_done = False
home_mode = False
stable_count = 0
STABLE_REQUIRED = 3
last_sent_command = ""
setting_target_mode = False
force_send = False

def log(msg):
    print(msg)

def compute_roundness(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0
    return 4 * np.pi * area / (perimeter * perimeter)

def detect_best_yellow_circle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask = cv2.GaussianBlur(mask, (7, 7), 2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_THRESHOLD:
            continue
        roundness = compute_roundness(cnt)
        if roundness >= ROUNDNESS_THRESHOLD:
            score = area * roundness
            (x, y), _ = cv2.minEnclosingCircle(cnt)
            if score > best_score:
                best = (int(x), int(y))
                best_score = score
    return best

def detect_yolo_center(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb)
    for box in results[0].boxes.data:
        x1, y1, x2, y2, *_ = box.tolist()
        cx = int((x1 + x2) / 2)
        cy = FRAME_HEIGHT - int((y1 + y2) / 2)
        return (cx, cy)
    return None

def calculate_steps(from_point, to_point):
    dx = to_point[0] - from_point[0]
    dy = from_point[1] - to_point[1]
    dx_mm = abs(dx) * PIXEL_TO_MM_X
    dy_mm = abs(dy) * PIXEL_TO_MM_Y
    dx_steps = min(int(dx_mm * STEPS_PER_MM), MAX_STEPS)
    dy_steps = min(int(dy_mm * STEPS_PER_MM), MAX_STEPS)
    dix = 1 if dx > 0 else 0
    diy = 1 if dy > 0 else 0
    command = f"{dix},{diy},{dx_steps},{dy_steps}"
    return dx, dy, dx_steps, dy_steps, command

class MainWindow(QWidget):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.setWindowTitle("Wafer Align (PyQt5)")
        self.setMinimumSize(640, 480)

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(3, FRAME_WIDTH)
        self.cap.set(4, FRAME_HEIGHT)

        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.label_coords = QLabel("남은좌표: (0, 0)")
        self.label_accuracy = QLabel("정확도: 0%")
        self.label_state = QLabel("상태: 대기")
        for label in [self.label_coords, self.label_accuracy, self.label_state]:
            label.setStyleSheet("font-size: 14px; padding: 4px;")

        status_layout = QHBoxLayout()
        status_layout.addWidget(self.label_coords)
        status_layout.addWidget(self.label_accuracy)
        status_layout.addWidget(self.label_state)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(status_layout)
        self.setLayout(layout)

        self.ser_reader = None
        self.ser_writer = None
        self.serial_ready = False
        self.command_task = None
        self.reset_timer = QTimer()
        self.reset_timer.setSingleShot(True)
        self.reset_timer.timeout.connect(self.set_waiting)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.setMouseTracking(True)
        self.show()

    def set_waiting(self):
        self.label_state.setText("상태: 정렬대기")

    async def serial_setup(self):
        try:
            self.ser_reader, self.ser_writer = await serial_asyncio.open_serial_connection(url=SERIAL_PORT, baudrate=BAUDRATE)
            self.serial_ready = True
            log("[INIT] Serial ready")
        except Exception as e:
            log(f"[ERR] Serial setup failed: {e}")
    async def send_serial_command(self, command):
        global sending_command, awaiting_done, last_sent_command, force_send, home_mode, send_enabled

        if not self.serial_ready or self.ser_writer is None:
            return
        if sending_command or awaiting_done:
            return
        if command == last_sent_command and not force_send:
            return

        sending_command = True
        awaiting_done = True
        try:
            self.ser_writer.write((command + "\n").encode())
            await self.ser_writer.drain()
            log(f"[SEND] {command}")
            last_sent_command = command
            force_send = False
            if home_mode:
                self.label_state.setText("상태: 원위치 중")
            else:
                self.label_state.setText("상태: 정렬중")

            while True:
                try:
                    line = await asyncio.wait_for(self.ser_reader.readline(), timeout=10.0)
                    decoded = line.decode().strip()
                    log(f"[ARDUINO] {decoded}")
                    if "DONE" in decoded or "READY" in decoded:
                        break
                except asyncio.TimeoutError:
                    log("[TIMEOUT] No DONE received")
                    break
        except Exception as e:
            log(f"[ERR] Serial: {e}")
        finally:
            sending_command = False
            awaiting_done = False
            if home_mode:
                self.label_state.setText("상태: 원위치 완료")
                home_mode = False
                send_enabled = False
            else:
                self.reset_timer.start(5000)

    def keyPressEvent(self, event):
        global confirmed_center, stable_count, last_sent_command
        global Target_Point, home_mode, send_enabled, setting_target_mode, force_send

        if not self.serial_ready:
            return

        key = event.key()
        if key == Qt.Key_S:
            send_enabled = not send_enabled
            log(f"[SEND MODE] {'Enabled' if send_enabled else 'Disabled'}")
            self.label_state.setText("상태: 정렬중" if send_enabled else "상태: 대기")
        elif key == Qt.Key_P:
            asyncio.create_task(self.send_serial_command("0,0,0,0"))
        elif key == Qt.Key_R:
            confirmed_center = None
            stable_count = 0
            last_sent_command = ""
        elif key == Qt.Key_U:
            if confirmed_center and not awaiting_done and not sending_command:
                home_mode = True
                dx, dy, dx_steps, dy_steps, command = calculate_steps(confirmed_center, HOME_DXDY)
                if dx_steps + dy_steps < 10:
                    self.label_state.setText("상태: 원위치 완료")
                    send_enabled = False
                else:
                    asyncio.create_task(self.send_serial_command(command))
        elif key == Qt.Key_M:
            setting_target_mode = not setting_target_mode
            log("[TARGET] Click to set Target_Point" if setting_target_mode else "[TARGET] Reset to default")
            if not setting_target_mode:
                Target_Point[0] = FRAME_WIDTH // 2
                Target_Point[1] = FRAME_HEIGHT // 2
        elif key == Qt.Key_Q:
            self.close()

    def mousePressEvent(self, event):
        global Target_Point, setting_target_mode, last_sent_command, force_send, confirmed_center
        if setting_target_mode and event.button() == Qt.LeftButton:
            x = int(event.pos().x() * FRAME_WIDTH / self.image_label.width())
            y = int(event.pos().y() * FRAME_HEIGHT / self.image_label.height())
            Target_Point = [x, y]
            last_sent_command = ""
            force_send = True
            setting_target_mode = False
            if confirmed_center:
                dx, dy, dx_steps, dy_steps, command = calculate_steps(confirmed_center, Target_Point)
                asyncio.create_task(self.send_serial_command(command))

    def update_frame(self):
        global confirmed_center, stable_count
        if not self.serial_ready:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        display = frame.copy()
        center = detect_best_yellow_circle(frame) or detect_yolo_center(frame)
        if center:
            confirmed_center = center
            dx, dy, dx_steps, dy_steps, command = calculate_steps(center, Target_Point)

            self.label_coords.setText(f"남은좌표: ({dx}, {dy})")

            if abs(dx) <= TOLERANCE_PX and abs(dy) <= TOLERANCE_PX:
                if send_enabled and not awaiting_done and not sending_command:
                    stable_count += 1
                if stable_count >= STABLE_REQUIRED:
                    self.label_state.setText("상태: 정렬완료")
            else:
                stable_count = 0

            if send_enabled and stable_count < STABLE_REQUIRED:
                asyncio.create_task(self.send_serial_command(command))

            if max(abs(dx), abs(dy)) < 100:
                error_ratio = max(abs(dx), abs(dy)) / max(FRAME_WIDTH, FRAME_HEIGHT)
                progress = max(0, 100 - int(error_ratio * 100))
            else:
                progress = 0
            self.label_accuracy.setText(f"정확도: {progress}%")
        else:
            self.label_coords.setText("남은좌표: (0, 0)")
            self.label_accuracy.setText("정확도: 0%")

        cv2.circle(display, tuple(Target_Point), 5, (0, 0, 255), -1)
        if confirmed_center:
            cv2.circle(display, confirmed_center, 5, (0, 255, 0), -1)
            cv2.line(display, tuple(Target_Point), confirmed_center, (255, 0, 0), 2)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(img))

def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = MainWindow(loop)
    loop.run_until_complete(window.serial_setup())
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    main()
