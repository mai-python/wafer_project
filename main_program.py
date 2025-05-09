import sys
import cv2
import numpy as np
import asyncio
import serial_asyncio
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy,
    QPushButton, QTextEdit, QMessageBox,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
from qasync import QEventLoop
import os
from PyQt5.QtWidgets import QInputDialog

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

#변수
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
Target_Point = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]
TOLERANCE_PX = 1
PIXEL_TO_MM_X = 0.6
PIXEL_TO_MM_Y = 0.83
STEPS_PER_MM = 55
MAX_STEPS = 32000
BAUDRATE = 9600
SERIAL_PORT = "/dev/ttyACM0"
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

Target_degree = 0
angle = 0

DEGREE_TO_STEP = 1
MAX_ROTATION_STEP = 32000

CLASS_COLORS = {
    'F1': (0, 0, 50),
    'F2': (0, 0, 100),
    'P100': (0, 0, 255),
    'P111': (255, 0, 0),
}

model = YOLO("runs/detect/train/weights/best.pt")

confirmed_center = None
send_enabled = False
sending_command = False
awaiting_done = False
home_mode = False
stable_count = 0
last_sent_command = ""
setting_target_mode = False
force_send = False
auto_mode = False

wafer_type_override = None
has_f1 = False
has_f2 = False
corrected_box = []

def show_warning(self, title, message):
    warning = QMessageBox(self)
    warning.setIcon(QMessageBox.Warning)
    warning.setWindowTitle(title)
    warning.setText(message)
    warning.setStandardButtons(QMessageBox.Ok)
    warning.exec_()

def calculate_accuracy_px(target_point, center_point):
    dx = target_point[0] - center_point[0]
    dy = target_point[1] - center_point[1]
    error_distance = (dx ** 2 + dy ** 2) ** 0.5

    MAX_ERROR_PX = 30        
    PERFECT_THRESH_PX = 1.5   

    if error_distance <= PERFECT_THRESH_PX:
        return 100.0
    elif error_distance >= MAX_ERROR_PX:
        return 0.0
    else:
        return round((1 - (error_distance / MAX_ERROR_PX)) * 100, 2)

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

def draw_yolo_boxes(frame, results, window=None, override_type=None):
    for box in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        class_name = results[0].names[int(cls)]

        if class_name not in ["F1", "F2"]:
            continue  # F1, F2만 시각화

        color = CLASS_COLORS.get(class_name, (0, 255, 0))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_name}"
        ((tw, th), _) = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(frame, (int(x1), int(y1 - th - 4)), (int(x1 + tw), int(y1)), color, -1)
        cv2.putText(frame, label, (int(x1), int(y1 - 4)), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

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

def calculate_rotation(current_angle, target_angle):
    dR = (target_angle - current_angle +540) % 360-180
    diR = 1 if dR > 0 else 0
    stR = min(int(abs(dR) * DEGREE_TO_STEP), MAX_ROTATION_STEP)
    return diR, stR

def calculate_steps(from_point, to_point):
    dx = to_point[0] - from_point[0]
    dy = from_point[1] - to_point[1]
    dx_mm = abs(dx) * PIXEL_TO_MM_X
    dy_mm = abs(dy) * PIXEL_TO_MM_Y
    dx_steps = min(int(dx_mm * STEPS_PER_MM), MAX_STEPS)
    dy_steps = min(int(dy_mm * STEPS_PER_MM), MAX_STEPS)
    dix = 1 if dx > 0 else 0
    diy = 1 if dy > 0 else 0
    command = f"{dix},{diy},{dx_steps},{dy_steps},0,0"
    return dx, dy, dx_steps, dy_steps, command

class MainWindow(QWidget):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.setWindowTitle("Wafer_Aligner")
        self.setMinimumSize(800, 600)

        self.warned_once = False

        self.zoom_window = ZoomWindow()

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

        self.s_button = QPushButton("ALIGN")
        self.m_button = QPushButton("MANUAL")
        self.u_button = QPushButton("HOMING")
        self.a_button = QPushButton("ANGLE")
        self.r_button = QPushButton("ROTATION")
        self.t_button = QPushButton("AUTO")

        self.s_button.clicked.connect(self.toggle_send)
        self.m_button.clicked.connect(self.toggle_target_mode)
        self.u_button.clicked.connect(self.move_home)
        self.a_button.clicked.connect(self.set_target_degree)
        self.r_button.clicked.connect(lambda: asyncio.create_task(self.start_rotation()))
        self.t_button.clicked.connect(self.toggle_auto_mode)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.s_button)
        button_layout.addWidget(self.m_button)
        button_layout.addWidget(self.u_button)
        button_layout.addWidget(self.a_button)
        button_layout.addWidget(self.r_button)
        button_layout.addWidget(self.t_button)

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
        send_enabled = True
        self.warned_once = False
        log("[SEND MODE] Enabled", self)
        self.label_state.setText("상태: 정렬중")

    def toggle_auto_mode(self):
        global auto_mode
        auto_mode = not auto_mode
        if auto_mode:
            log("[AUTO] 시작됨", self)
            asyncio.create_task(self.auto_align_loop())
        else:
            log("[AUTO] 중단됨", self)     
    
    def set_target_degree(self):
        degree, ok = QInputDialog.getDouble(self, "SETTING ", "INPUT", decimals=1,min=0,max=360)
        if ok:
            global Target_degree
            Target_degree = degree
            log(f"[ROTATION] 목표 각도 설정됨: {Target_degree:.1f}도", self)

    async def start_rotation(self):
            global angle, Target_degree
            self.label_state.setText("상태: 회전중")

            for _ in range(10):
                await asyncio.sleep(0.1)
                diR, stR = calculate_rotation(angle, Target_degree)

                if stR < 5:
                    log(f"[ROTATION] COMPLETE: 오차 {abs(Target_degree - angle):.2f}도", self)
                    self.label_state.setText("상태: 회전 완료")
                    return
                command = f"0,0,0,0,{diR},{stR}"
                await self.send_serial_command(command)

    async def auto_align_loop(self):
        global auto_mode, send_enabled, home_mode, force_send

        while auto_mode:
            if not auto_mode:
                break
            if confirmed_center:
                home_mode = True
                dx, dy, dx_steps, dy_steps, command = calculate_steps(confirmed_center, HOME_DXDY)
                if dx_steps + dy_steps >= 10:
                    await self.send_serial_command(command)
                    await asyncio.sleep(0.5)

            send_enabled = True
            stable_count = 0
            self.label_state.setText("상태: 정렬중")

            for _ in range(100):  
                await asyncio.sleep(0.05)
                if stable_count >= STABLE_REQUIRED:
                    break

            send_enabled = False
            await self.send_serial_command("STOP")
            self.label_state.setText("상태: 정렬완료")
            await asyncio.sleep(0.5)

        auto_mode = False
        send_enabled = False
        force_send = False
        self.label_state.setText("state: standby")
        log("[AUTO] PAUSED", self)

    def toggle_target_mode(self):
        global setting_target_mode
        setting_target_mode = not setting_target_mode
        log("[TARGET] Click to set Target_Point" if setting_target_mode else "[TARGET] Reset to default", self)

    def move_home(self):
        global confirmed_center, home_mode, dx, dy, send_enabled
        if confirmed_center and not awaiting_done and not sending_command:
            home_mode = True
            dx, dy, dx_steps, dy_steps, command = calculate_steps(confirmed_center, HOME_DXDY)
            if dx_steps + dy_steps < 10:
                self.label_state.setText("상태: 원위치 완료")
                send_enabled = False
            else:
                asyncio.create_task(self.send_serial_command(command))

    def mousePressEvent(self, event):
        global Target_Point, setting_target_mode, last_sent_command, force_send, confirmed_center, send_enabled
        if setting_target_mode and event.button() == Qt.LeftButton:
            x = int(event.pos().x() * FRAME_WIDTH / self.image_label.width())
            y = int(event.pos().y() * FRAME_HEIGHT / self.image_label.height())
            Target_Point = [x, y]
            log(f"[TARGET] Target set to: {Target_Point}", self)
            last_sent_command = ""
            force_send = True
            setting_target_mode = False

            send_enabled = False
            self.label_state.setText("상태: 대기")

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
        wafer_type_override = None
# --- draw_yolo_boxes() 호출 전 F1/F2 존재 여부 판단
        has_f1 = False
        has_f2 = False

        for box in results[0].boxes.data:
            _, _, _, _, _, cls = box.tolist()
            label = results[0].names[int(cls)]
            if label == "F1":
                has_f1 = True
            elif label == "F2":
                has_f2 = True

        draw_yolo_boxes(display, results, self, wafer_type_override)
        print("[YOLO DETECTED]", [results[0].names[int(cls)] for *_, cls in results[0].boxes.data.tolist()])
        print("[CLASSES]", results[0].names)


 # ---- F1/F2 중심좌표 계산 (루프 안에서)
        wafer_center = None
        f1_center = None
        f2_center = None

        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            label = results[0].names[int(cls)]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if label in ["P100", "P111"]:
                wafer_center = (cx, cy)
            elif label == "F1":
                f1_center = (cx, cy)
            elif label == "F2":
                f2_center = (cx, cy)

        # ---- 🔧 루프 바깥에서 시각화 처리
        if has_f1 and has_f2 and f1_center and f2_center:
            angle_f1f2 = calculate_angle(f1_center, f2_center)
            angle_text = f"{angle_f1f2:.2f}°"
            text_pos = (f2_center[0] + 10, f2_center[1] - 10)
            cv2.putText(display, angle_text, text_pos, FONT, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)
            cv2.line(display, f1_center, f2_center, (255, 0, 0), 2)
            cv2.circle(display, f1_center, 5, (0, 255, 0), -1)
            cv2.circle(display, f2_center, 5, (0, 0, 255), -1)

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
                    asyncio.create_task(self.send_serial_command("STOP"))
            else:
                stable_count = 0
            if send_enabled and stable_count < STABLE_REQUIRED:
                
                asyncio.create_task(self.send_serial_command(command))

            if confirmed_center:
                accuracy = calculate_accuracy_px(Target_Point, confirmed_center)
                self.label_accuracy.setText(f"정확도: {accuracy}%")
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

        zoom_size = 100
        half = zoom_size // 2
        cx, cy = Target_Point

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(display.shape[1], cx + half)
        y2 = min(display.shape[0], cy + half)

        zoom_crop = display[y1:y2, x1:x2]

        zoomed = cv2.resize(zoom_crop, (200, 200), interpolation=cv2.INTER_NEAREST)
        zoomed_rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
        self.zoom_window.update_zoom(zoomed_rgb)

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

class ZoomWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZOOM_POINT")
        self.setFixedSize(500, 400)

        self.zoom_label = QLabel()
        self.zoom_label.setScaledContents(True)

        layout = QVBoxLayout()
        layout.addWidget(self.zoom_label)
        self.setLayout(layout)
        self.show()

    def update_zoom(self, image):
        h, w, ch = image.shape
        qimg = QImage(image.data, w, h, ch * w, QImage.Format_RGB888)
        self.zoom_label.setPixmap(QPixmap.fromImage(qimg))

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
