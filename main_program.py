import sys
import cv2
import numpy as np
import asyncio
import serial_asyncio
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy,
    QPushButton, QTextEdit, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
from qasync import QEventLoop
import os

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
Target_Point = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]
TOLERANCE_PX = 5
PIXEL_TO_MM_X = 0.6
PIXEL_TO_MM_Y = 0.83
STEPS_PER_MM = 55
MAX_STEPS = 32000
BAUDRATE = 9600
SERIAL_PORT = "/dev/ttyACM1"
CAMERA_INDEX = 0
HOME_DXDY = (400, 200)

YELLOW_LOWER = (15, 100, 100)
YELLOW_UPPER = (40, 255, 255)
AREA_THRESHOLD = 1000
ROUNDNESS_THRESHOLD = 0.4
STABLE_REQUIRED = 3

FONT_SCALE = 0.6
FONT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (255, 255, 255)

CLASS_COLORS = {
    'F1': (0, 255, 255),
    'F2': (0, 255, 0),
    'P100': (0, 0, 255),
    'P111': (255, 0, 0),
}
model = YOLO("runs/detect/train4/weights/best.pt")

confirmed_center = None
send_enabled = False
sending_command = False
awaiting_done = False
home_mode = False
stable_count = 0
last_sent_command = ""
setting_target_mode = False
force_send = False

def log(msg, window):
    print(msg)
    window.log_box.append(msg)

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

def draw_yolo_boxes(frame, results, window=None):
    for box in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        class_name = results[0].names[int(cls)]
        color = CLASS_COLORS.get(class_name, (0, 255, 0))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_name}"
        ((tw, th), _) = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(frame, (int(x1), int(y1 - th - 4)), (int(x1 + tw), int(y1)), color, -1)
        cv2.putText(frame, label, (int(x1), int(y1 - 4)), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        if window:
            log(f"[DETECT] {class_name} at ({int((x1 + x2) / 2)}, {int((y1 + y2) / 2)})", window)

def detect_yolo_center(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb)
    for box in results[0].boxes.data:
        x1, y1, x2, y2, *_ = box.tolist()
        cx = int((x1 + x2) / 2)
        cy = FRAME_HEIGHT - int((y1 + y2) / 2)
        return (cx, cy)
    return None

def calculate_angle(pt1, pt2):
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle + 360 if angle < 0 else angle

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
        self.setWindowTitle("Wafer_Aligner")
        self.setMinimumSize(800, 600)

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

        self.s_button = QPushButton("정렬시작")
        self.m_button = QPushButton("목표설정")
        self.u_button = QPushButton("원위치")
        self.r_button = QPushButton("리셋")
        self.s_button.clicked.connect(self.toggle_send)
        self.m_button.clicked.connect(self.toggle_target_mode)
        self.u_button.clicked.connect(self.move_home)
        self.r_button.clicked.connect(self.reset)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.s_button)
        button_layout.addWidget(self.m_button)
        button_layout.addWidget(self.u_button)
        button_layout.addWidget(self.r_button)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(status_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self.log_box)
        self.setLayout(layout)

        self.ser_reader = None
        self.ser_writer = None
        self.serial_ready = False
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
            self.ser_reader, self.ser_writer = await serial_asyncio.open_serial_connection(
                url=SERIAL_PORT, baudrate=BAUDRATE)
            self.serial_ready = True
            log("[INIT] Serial ready", self)
        except Exception as e:
            log(f"[ERR] Serial setup failed: {e}", self)

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
            log(f"[SEND] {command}", self)
            last_sent_command = command
            force_send = False
            self.label_state.setText("상태: 원위치 중" if home_mode else "상태: 정렬중")

            while True:
                try:
                    line = await asyncio.wait_for(self.ser_reader.readline(), timeout=10.0)
                    decoded = line.decode().strip()
                    log(f"[ARDUINO] {decoded}", self)
                    if "DONE" in decoded or "READY" in decoded:
                        break
                except asyncio.TimeoutError:
                    log("[TIMEOUT] No DONE received", self)
                    break
        except Exception as e:
            log(f"[ERR] Serial: {e}", self)
        finally:
            sending_command = False
            awaiting_done = False
            if home_mode:
                self.label_state.setText("상태: 원위치 완료")
                home_mode = False
                send_enabled = False
            else:
                self.reset_timer.start(5000)

    def toggle_send(self):
        global send_enabled
        send_enabled = not send_enabled
        log(f"[SEND MODE] {'Enabled' if send_enabled else 'Disabled'}", self)
        self.label_state.setText("상태: 정렬중" if send_enabled else "상태: 대기")

    def toggle_target_mode(self):
        global setting_target_mode
        setting_target_mode = not setting_target_mode
        log("[TARGET] Click to set Target_Point" if setting_target_mode else "[TARGET] Reset to default", self)

    def move_home(self):
        global confirmed_center, home_mode
        if confirmed_center and not awaiting_done and not sending_command:
            home_mode = True
            dx, dy, dx_steps, dy_steps, command = calculate_steps(confirmed_center, HOME_DXDY)
            if dx_steps + dy_steps < 10:
                self.label_state.setText("상태: 원위치 완료")
                send_enabled = False
            else:
                asyncio.create_task(self.send_serial_command(command))

    def mousePressEvent(self, event):
        global Target_Point, setting_target_mode, last_sent_command, force_send, confirmed_center
        if setting_target_mode and event.button() == Qt.LeftButton:
            x = int(event.pos().x() * FRAME_WIDTH / self.image_label.width())
            y = int(event.pos().y() * FRAME_HEIGHT / self.image_label.height())
            Target_Point = [x, y]
            log(f"[TARGET] Target set to: {Target_Point}", self)
            last_sent_command = ""
            force_send = True
            setting_target_mode = False

    def reset(self):
        global confirmed_center, stable_count, last_sent_command
        confirmed_center = None
        stable_count = 0
        last_sent_command = ""

    def update_frame(self):
        global confirmed_center, stable_count
        if not self.serial_ready:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        display = frame.copy()
        results = model(display)
        draw_yolo_boxes(display, results, self)

        wafer_center = None
        f1_center = None
        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            label = results[0].names[int(cls)]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if label in ["P100", "P111"]:
                wafer_center = (cx, cy)
            elif label == "F1":
                f1_center = (cx, cy)

        if wafer_center and f1_center:
            angle = calculate_angle(wafer_center, f1_center)
            log(f"[ANGLE] {angle:.2f} deg (Wafer to F1)", self)
            angle_text = f"{angle:.2f}deg"
            text_pos = (wafer_center[0] + 10, wafer_center[1] - 10)
            cv2.putText(display, angle_text, text_pos, FONT, FONT_SCALE, (255, 255, 0), FONT_THICKNESS)

        center = detect_best_yellow_circle(display) or detect_yolo_center(display)
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

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F1:
            self.show_help_dialog()

    def show_help_dialog(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("도움말")
        msg.setText("Wafer Aligner\n\n정렬시작: 정렬을 시작합니다\n목표설정: 마우스로 타겟 설정\n원위치: 초기 위치로 복귀\n리셋: 상태 초기화")
        msg.exec_()

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
