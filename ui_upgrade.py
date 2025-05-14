import sys, os, gc, time, asyncio, cv2, torch, numpy as np, serial_asyncio
from ultralytics import YOLO
from qasync import QEventLoop
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit, QInputDialog,
    QVBoxLayout, QHBoxLayout, QFormLayout, QScrollArea, QComboBox, QSizePolicy, QLineEdit, QMessageBox, QTabWidget,
    QAction, QMenu, QSplashScreen, QGridLayout, QProgressBar
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
import matplotlib.pyplot as plt
model = YOLO("C:\\Users\\Mai\\Desktop\\wafer_project\\best.pt") # best.pt check
##########################################수정금지#########################################################

FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
Target_Point = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]
HOME_DXDY = (400, 200)

TOLERANCE_PX, TOLERANCE_R = 1, 1
PIXEL_TO_MM_X, PIXEL_TO_MM_Y = 0.6, 0.83
STEPS_PER_MM, MAX_STEPS = 55, 32000

CAMERA_INDEX = 0
BAUDRATE = 9600
SERIAL_PORT = "COM6"

YELLOW_LOWER, YELLOW_UPPER = (15, 100, 100), (40, 255, 255)
AREA_THRESHOLD, ROUNDNESS_THRESHOLD = 1000, 0.4
STABLE_REQUIRED = 3

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE, FONT_THICKNESS = 0.6, 2
TEXT_COLOR = (255, 255, 255)

Target_degree, angle = 0, 0
DEGREE_TO_STEP, MAX_ROTATION_STEP = 1, 32000

CLASS_COLORS = {
    'F1': (0, 0, 50), 'F2': (0, 0, 100),
    'P100': (0, 0, 255), 'P111': (255, 0, 0),
}

detection_stats = {
    "P100": {"F1": [], "F2": [], "P100": [], "P111": []}, #png저장용
    "P111": {"F1": [], "F2": [], "P100": [], "P111": []}
}

confirmed_center = None
send_enabled = sending_command = awaiting_done = home_mode = False
stable_count = 0
last_sent_command = ""
setting_target_mode = force_send = auto_mode = False
wafer_type_override = None
has_f1 = has_f2 = False
fixed_f1_box = fixed_f2_box = None
corrected_box = []
#######################################################################################################
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
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    full_msg = f"{timestamp} {msg}"
    print(full_msg)
    if hasattr(window, 'log_window') and window.log_window:
        window.log_window.append_log(full_msg)

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
            continue

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
    dR = (target_angle - current_angle + 540) % 360 - 180
    diR = 1
    stR = max(1, min(int(abs(dR) * DEGREE_TO_STEP), MAX_ROTATION_STEP))
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
##########################################수정금지#########################################################
class MainWindow(QMainWindow):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.setWindowTitle("Wafer_Aligner")
        self.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT + 200 + self.menuBar().height())
        self.setMouseTracking(True)
        # 메뉴바
        menubar = self.menuBar()
        file_menu = menubar.addMenu("FILE")
        file_menu.addAction(QAction("EXIT", self, triggered=self.close))
        capture_action = QAction("Capture Screenshot", self)
        capture_action.triggered.connect(self.capture_screenshot)
        file_menu.addAction(capture_action)

        function_menu = menubar.addMenu("Function")
        function_menu.addAction(QAction("Align", self, triggered=self.show_align_tab))
        function_menu.addAction(QAction("Rotation", self, triggered=self.show_angle_tab))
        function_menu.addAction(QAction("Advanced", self, triggered=self.show_advanced_tab))

        tool_menu = menubar.addMenu("Tool")
        settings_menu = QMenu("Settings", self)
        tool_menu.addAction(QAction("RELOAD_MODE_YOLO", self, triggered=self.reload_yolo_model))
        tool_menu.addMenu(settings_menu)    
        settings_menu.addAction("Show Graph", self.show_graph_tab)
        settings_menu.addAction("Variables", self.show_variable_tab)
        help_menu = menubar.addMenu("Help")

        help_menu.addAction(QAction("Guide", self, triggered=self.show_help_dialog))

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(3, FRAME_WIDTH)
        self.cap.set(4, FRAME_HEIGHT)

        self.zoom_window = None
        self.log_window = LogWindow()
        self.rotation_active = False
        self.warned_once = False

        self.graph_tab = QWidget()
        graph_layout = QVBoxLayout()
        self.graph_canvas = GraphCanvas(self)
        graph_layout.addWidget(self.graph_canvas)
        self.graph_tab.setLayout(graph_layout)

        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.hide()
        self.setup_tabs()

        self.label_message = QLabel(" 대기중 ")
        self.label_message.setStyleSheet("font-size: 16px; color: black;")

        self.align_label = QLabel("ALIGN_PROGRESS")
        self.align_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.rotation_label = QLabel("ROTATION_PROGRESS")
        self.rotation_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        self.align_bar = QProgressBar()
        self.rotation_bar = QProgressBar()
        self.align_bar.setMaximum(100)
        self.rotation_bar.setMaximum(100)
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(self.label_message)
        progress_layout.addWidget(self.align_label)
        progress_layout.addWidget(self.align_bar)
        progress_layout.addWidget(self.rotation_label)
        progress_layout.addWidget(self.rotation_bar)
        self.progress_widget = QWidget()
        self.progress_widget.setLayout(progress_layout)
        self.progress_widget.setFixedWidth(300)

        self.label_wafer_type = QLabel("Wafer Type: -")
        self.label_coords = QLabel("Offset: (0, 0)")
        self.label_accuracy = QLabel("Accuracy: 0%")
        self.label_state = QLabel("Status: Standby")
        self.label_angle = QLabel("Angle: 0.00°")
        self.label_angle_error = QLabel("Angle_Err: 0.00°")
        for lbl in [self.label_wafer_type, self.label_coords, self.label_accuracy,
                    self.label_state, self.label_angle, self.label_angle_error]:
            lbl.setStyleSheet("font-size: 15px; padding: 10px;")

        status_layout = QGridLayout()
        status_layout.addWidget(self.label_wafer_type, 0, 0)
        status_layout.addWidget(self.label_coords, 0, 1)
        status_layout.addWidget(self.label_accuracy, 0, 2)
        status_layout.addWidget(self.label_state, 1, 0)
        status_layout.addWidget(self.label_angle, 1, 1)
        status_layout.addWidget(self.label_angle_error, 1, 2)

        self.status_widget = QWidget()
        self.status_widget.setLayout(status_layout)
        self.status_widget.setFixedWidth(460)

        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.tabs)
        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(520)

        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)
        bottom_layout.addWidget(left_panel)
        bottom_layout.addWidget(self.progress_widget)
        bottom_layout.addWidget(self.status_widget)

        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_layout)
        bottom_widget.setFixedHeight(200)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(bottom_widget)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.initial_offset = (100, 100)
        self.initial_angle = 140
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

    def setup_tabs(self):
        self.align_tab = QWidget()
        layout = QVBoxLayout()
        self.s_button = QPushButton("ALIGN")
        self.m_button = QPushButton("MANUAL")
        self.u_button = QPushButton("HOMING")
        for btn in [self.s_button, self.m_button, self.u_button]:
            btn.setFixedHeight(50)
            btn.setFixedWidth(460)
            btn.setStyleSheet("font-size: 20px; padding: 6px;")
            layout.addWidget(btn)
        self.align_tab.setLayout(layout)
        self.s_button.clicked.connect(self.toggle_send)
        self.m_button.clicked.connect(self.toggle_target_mode)
        self.u_button.clicked.connect(self.move_home)

        self.angle_tab = QWidget()
        layout = QVBoxLayout()
        self.a_button = QPushButton("ANGLE")
        self.r_button = QPushButton("ROTATION")
        for btn in [self.a_button, self.r_button]:
            btn.setFixedHeight(70)
            btn.setFixedWidth(460)
            btn.setStyleSheet("font-size: 20px; padding: 6px;")
            layout.addWidget(btn)
        self.angle_tab.setLayout(layout)
        self.a_button.clicked.connect(self.set_target_degree)
        self.r_button.clicked.connect(lambda: asyncio.create_task(self.start_rotation()))

        self.advanced_tab = QWidget()
        layout = QVBoxLayout()
        self.t_button = QPushButton("AUTO")
        self.t_button.setFixedHeight(70)
        self.t_button.setFixedWidth(460)
        self.t_button.setStyleSheet("font-size: 20px; padding: 6px;")
        layout.addWidget(self.t_button)
        self.advanced_tab.setLayout(layout)
        self.t_button.clicked.connect(lambda: asyncio.create_task(self.auto_align_loop()))

        self.variable_tab = VariableTab(self) #variable tab
############################################################################################################
    def show_align_tab(self):
        if self.tabs.indexOf(self.align_tab) == -1:
            self.tabs.addTab(self.align_tab, "Align")
        self.tabs.show()
        self.tabs.setCurrentWidget(self.align_tab)

    def show_angle_tab(self):
        if self.tabs.indexOf(self.angle_tab) == -1:
            self.tabs.addTab(self.angle_tab, "Angle")
        self.tabs.show()
        self.tabs.setCurrentWidget(self.angle_tab)

    def show_advanced_tab(self):
        if self.tabs.indexOf(self.advanced_tab) == -1:
            self.tabs.addTab(self.advanced_tab, "Advanced")
        self.tabs.show()
        self.tabs.setCurrentWidget(self.advanced_tab)

    def update_progress(self, dx, dy, angle_error):
        ox, oy = self.initial_offset
        remain_dist = (abs(dx) + abs(dy)) / 2
        full_dist = (ox + oy) / 2
        align_progress = max(0, min(1, 1 - remain_dist / full_dist)) * 100

        rotation_progress = max(0, min(1, 1 - angle_error / self.initial_angle)) * 100

        self.align_bar.setValue(int(align_progress))
        self.rotation_bar.setValue(int(rotation_progress))
    def set_target_degree(self):
        degree, ok = QInputDialog.getDouble(self, "SETTING", "INPUT DEGREE", decimals=1, min=0, max=360)
        if ok:
            global Target_degree
            Target_degree = degree
            log(f"[ROTATION] Set Target Degree: {Target_degree:.1f}°", self)
            self.label_message.setText("목표 각도 설정됨")
        
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Exit", "Do you want to save the log and summary before exiting?", QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if reply == QMessageBox.Yes:
            if hasattr(self, 'log_window') and self. log_window:
                self.log_window.save_log_txt()
            if self.log_window.isVisible():
                self.log_window.close()
            event.accept()
        elif reply == QMessageBox.No:
            event.accept()
            if self.log_window.isVisible():
                self.log_window.close()
            event.accept()
        else:
            event.ignore()

    def set_waiting(self):
        self.label_state.setText("Status: ALIGN_STANDBY")

    def show_graph_tab(self):
        if self.tabs.indexOf(self.graph_tab) == -1:
            self.tabs.addTab(self.graph_tab, "Graph")
        self.tabs.show()
        self.tabs.setCurrentWidget(self.graph_tab)

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
            if command == "STOP":
                command = "0,0,0,0,0,0" 
            
            self.ser_writer.write((command + "\n").encode())
            await self.ser_writer.drain()
            log(f"[SEND] {command}", self)
            last_sent_command = command
            force_send = False
            self.label_state.setText("Status: HOMING" if home_mode else "Status: ALIGNING")

            while True:
                try:
                    line = await asyncio.wait_for(self.ser_reader.readline(), timeout=30.0)
                    decoded = line.decode().strip()
                    log(f"[ARDUINO] {decoded}", self)
                    if "DONE" in decoded or "READY" in decoded:
                        break
                except asyncio.TimeoutError:
                    log("[TIMEOUT] No DONE received", self)
                    self.label_state.setText("Status: ALIGN_STANDBY")
                    break
        except Exception as e:
            log(f"[ERR] Serial: {e}", self)
        finally:
            sending_command = False
            awaiting_done = False
            if home_mode:
                self.label_state.setText("Status: HOMING_DONE")
                home_mode = False
                send_enabled = False
            else:
                log("[ROTATION] ROTATION_DONE", self)
                self.label_state.setText("Status: ROTATION_DONE")
                self.reset_timer.start(5000)

    def toggle_send(self):
        global send_enabled
        send_enabled = not send_enabled
        log(f"[SEND MODE] {'Enabled' if send_enabled else 'Disabled'}", self)
        self.label_state.setText("Status: ALIGNING" if send_enabled else "Status: ALIGN_STANDBY")
        send_enabled = True
        self.warned_once = False
        log("[SEND MODE] Enabled", self)
        self.label_message.setText("정렬 중")
        self.label_state.setText("Status: ALIGNING")

    async def start_rotation(self):
        global angle, Target_degree, angle
        self.rotation_active = True
        self.label_message.setText("잠시 후 정렬 시작")
        self.label_state.setText("Status: ROTATIONING")

        while True:
            await asyncio.sleep(0.2)

            ret, frame = self.cap.read()
            if not ret:
                continue

            results = model(frame)
            f1_center, wafer_center = None, None

            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                label = results[0].names[int(cls)]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if label in ["P100", "P111"]:
                    wafer_center = (cx, cy)
                elif label == "F1":
                    f1_center = (cx, cy)

            if not f1_center or not wafer_center:
                log("[DEBUG] CAN'T FIND WAFER_CENTER → DO NOT ROTATE", self)
                continue

            angle = calculate_angle(f1_center, wafer_center)
            diff = (Target_degree - angle + 540) % 360 - 180
            angle_error = abs(diff)

            self.label_angle.setText(f"Angle: {angle:.2f}°")
            self.label_angle_error.setText(f"Angle_Err: {angle_error:.2f}°")

            log(f"[DEBUG] angle={angle:.2f}, target={Target_degree:.2f}, error={angle_error:.2f}", self)

            if angle_error < TOLERANCE_R: 
                log(f"[FORCE STOP] Angle_Err {angle_error:.2f}°", self)
                self.label_state.setText("Status: ROTATION_DONE")
                await self.send_serial_command("STOP")
                self.rotation_active = False
                self.update_points_after_rotation(frame)
                break  
            else:
                diR, stR = calculate_rotation(angle, Target_degree)
                command = f"0,0,0,0,{diR},{stR}"
                log(f"[DEBUG] Send Command: {command}", self)
                await self.send_serial_command(command)  

    def update_points_after_rotation(self, frame):
        results = model(frame)
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

        if wafer_center:
            self.confirmed_center = wafer_center 
            log(f"[UPDATED POINTS] confirmed_center: {self.confirmed_center}", self)
            self.label_state.setText("Status: ALIGN_STANDBY")  

    def reload_yolo_model(self):
        global model
        log("[YOLO] reload model...", self)
        self.label_message.setText("모델 초기화 중")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = YOLO("best/v8m960best.pt")
        log("[YOLO]complete reload model", self)
        self.label_message.setText("모델 초기화 완료")

    async def auto_align_loop(self):
        global auto_mode, send_enabled, home_mode, force_send

        while auto_mode:
            self.label_message.setText("잠시 후 정렬 시작")
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
            self.label_state.setText("Status: ALIGNING")
            self.label_message.setText("정렬 중")

            for _ in range(100):  
                await asyncio.sleep(0.05)
                if stable_count >= STABLE_REQUIRED:
                    break

            send_enabled = False
            await self.send_serial_command("STOP")
            self.label_state.setText("Status: ALIGN_DONE")
            await asyncio.sleep(0.5)

        auto_mode = False
        send_enabled = False
        force_send = False
        self.label_state.setText("Status: Standby")
        log("[AUTO] PAUSED", self)

    def toggle_target_mode(self):
        global setting_target_mode
        setting_target_mode = not setting_target_mode
        log("[TARGET] Click to set Target_Point" if setting_target_mode else "[TARGET] Reset to default", self)

    def move_home(self):
        global confirmed_center, home_mode, dx, dy, send_enabled
        if confirmed_center and not awaiting_done and not sending_command:
            home_mode = True
            self.label_message.setText("정렬 중")
            dx, dy, dx_steps, dy_steps, command = calculate_steps(confirmed_center, HOME_DXDY)
            if dx_steps + dy_steps < 10:
                self.label_state.setText("Status: HOMING_DONE")
                send_enabled = False
            else:
                asyncio.create_task(self.send_serial_command(command))

    def mousePressEvent(self, event):
        global Target_Point, setting_target_mode, last_sent_command, force_send, confirmed_center, send_enabled

        if setting_target_mode and event.button() == Qt.LeftButton:
            pos = event.pos()
            widget_pos = self.image_label.mapFrom(self, pos)
            x = widget_pos.x()
            y = widget_pos.y()

            if 0 <= x < self.image_label.width() and 0 <= y < self.image_label.height():
                Target_Point = [int(x), int(y)]
                log(f"[TARGET] 정확 좌표 설정됨 → {Target_Point}", self)

                last_sent_command = ""
                force_send = True
                setting_target_mode = False
                send_enabled = False
                self.label_state.setText("Status: STANDBY")

    def reset(self):
        global confirmed_center, stable_count, last_sent_command
        confirmed_center = None
        stable_count = 0
        last_sent_command = ""

    def update_frame(self):
        global confirmed_center, stable_count, angle

        if not self.serial_ready:
            return

        ret, frame = self.cap.read()
        if not ret:
            if not self.warned_once:
                log("[CAMERA] Received failed to frame", self)
                self.warned_once = True
            return
        
        display = frame.copy()
        results = model(display, conf=0.4)

        if not self.rotation_active: 
            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                label = results[0].names[int(cls)]
                print(f"Detected: {label}, conf: {conf:.2f}")

        has_f1 = False
        has_f2 = False
        for box in results[0].boxes.data:
            _, _, _, _, _, cls = box.tolist()
            label = results[0].names[int(cls)]
            if label == "F1":
                has_f1 = True
            elif label == "F2":
                has_f2 = True

        wafer_type = None
        if has_f1 and has_f2:
            wafer_type = "P100"
        elif has_f1:
            wafer_type = "P111"

        draw_yolo_boxes(display, results, self, wafer_type)

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
            if wafer_type in ["P100", "P111"]:
                for box in results[0].boxes.data:
                    _, _, _, _, conf, cls = box.tolist()
                    label = results[0].names[int(cls)]
                    if label in detection_stats[wafer_type]:
                        detection_stats[wafer_type][label].append(conf)

        if f1_center and wafer_center:
            angle = calculate_angle(f1_center, wafer_center) 
            angle_text = f"{angle:.2f}°"
            self.label_angle.setText(f"Angle: {angle:.2f}°")

            diff = (Target_degree - angle + 540) % 360 - 180
            angle_error = abs(diff)
            self.label_angle_error.setText(f"Aangle_Err: {angle_error:.2f}°")

            cv2.putText(display, angle_text, (f1_center[0] + 10, f1_center[1] - 10), FONT, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)

        center = detect_best_yellow_circle(display) or detect_yolo_center(display)
        if center:
            confirmed_center = center
            dx, dy, dx_steps, dy_steps, command = calculate_steps(center, Target_Point)
            self.label_coords.setText(f"Offset: ({dx}, {dy})")

            if abs(dx) <= TOLERANCE_PX and abs(dy) <= TOLERANCE_PX:
                if send_enabled and not awaiting_done and not sending_command:
                    stable_count += 1
                if stable_count >= STABLE_REQUIRED:
                    self.label_state.setText("Status: DONE")
                    asyncio.create_task(self.send_serial_command("STOP"))
            else:
                stable_count = 0

            if send_enabled and stable_count < STABLE_REQUIRED:
                asyncio.create_task(self.send_serial_command(command))

            accuracy = calculate_accuracy_px(Target_Point, confirmed_center)
            self.graph_canvas.update_plot(accuracy)
            self.label_accuracy.setText(f"Accuracy: {accuracy}%")
        else:
            self.label_coords.setText("Offset: (0, 0)")
            self.label_accuracy.setText("Accuracy: 0%")

        if not self.rotation_active:  
            cv2.circle(display, tuple(Target_Point), 5, (0, 0, 255), -1)
            if confirmed_center:
                cv2.circle(display, confirmed_center, 5, (0, 255, 0), -1)
                cv2.line(display, tuple(Target_Point), confirmed_center, (255, 0, 0), 2)
        wafer_type = None
        if has_f1 and has_f2:
            wafer_type = "P100"
        elif has_f1:
            wafer_type = "P111"
        else:
            wafer_type = "-"

        self.label_wafer_type.setText(f"Wafer Type: {wafer_type}")

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(img)
        self.image_label.setPixmap(pixmap)

        zoom_size = 100
        half = zoom_size // 2
        cx, cy = Target_Point

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(display.shape[1], cx + half)
        y2 = min(display.shape[0], cy + half)

        zoom_crop = display[y1:y2, x1:x2]

        if zoom_crop.size > 0:
            zoomed = cv2.resize(zoom_crop, (400, 300), interpolation=cv2.INTER_NEAREST)
            zoomed_rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)

            if hasattr(self, 'zoom_window') and self.zoom_window:
                self.zoom_window.update_zoom(zoomed_rgb)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F1:
            self.show_help_dialog()
        elif key == Qt.Key_F2:
            self.toggle_zoom_window()
        elif key == Qt.Key_F3:
            if self.log_window.isVisible():
                self.log_window.hide()  
            else:
                self.log_window.show()  
        elif key == Qt.Key_F5:
            self.reload_yolo_model()
    
    def capture_screenshot(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        pixmap = self.grab()
        pixmap.save(filename)
        QMessageBox.information(self, "Screenshot", f"Saved as {filename}")

    def show_help_dialog(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("도움말")
        msg.setText("Wafer Aligner\n\nALIGN: 정렬을 시작합니다\nMANUAL: 마우스로 타겟 설정\nHOMING: 초기 위치로 복귀\nANGLE: 각도 설정 \nROTATION: 각도 정렬 \nAUTO: 자동모드 ")
        msg.exec_()
    def show_variable_tab(self):
        if self.tabs.indexOf(self.variable_tab) == -1:
            self.tabs.addTab(self.variable_tab, "Variables")
        self.tabs.show()
        self.tabs.setCurrentWidget(self.variable_tab)

    def toggle_zoom_window(self):
        if self.zoom_window is None:
            self.zoom_window = ZoomWindow()
        else:
            self.zoom_window.show()
            self.zoom_window.raise_()

class LogWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LOG")
        self.setGeometry(200, 200, 600, 400)

        self.logs = []

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        self.filter_box = QComboBox()
        self.filter_box.addItems(["ALL", "ALIGN", "ROTATION", "DEBUG", "CHANGED", "ERROR"]) #추후에 추가
        self.filter_box.currentTextChanged.connect(self.apply_filter)

        layout = QVBoxLayout()
        layout.addWidget(self.filter_box)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

    def append_log(self, msg):
        self.logs.append(msg)
        self.apply_filter()

    def apply_filter(self):
        selected = self.filter_box.currentText()
        self.text_edit.clear()
        for msg in self.logs:
            if selected == "ALL" or f"[{selected}]" in msg:
                self.text_edit.append(msg)
        self.text_edit.verticalScrollBar().setValue(
            self.text_edit.verticalScrollBar().maximum()
        )
    def save_detection_summary_chart(wafer_type, stats, timestamp): #png(yolo)저장함수
        classes = list(stats.keys())
        counts = [len(stats[cls]) for cls in classes]
        confidences = [
            round(sum(stats[cls]) / len(stats[cls]), 2) if stats[cls] else 0
            for cls in classes
        ]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.bar(classes, counts, label="Detection Count")
        ax1.set_ylabel("Count")
        ax1.set_ylim(0, max(counts) + 1)

        ax2 = ax1.twinx()
        ax2.plot(classes, confidences, marker="o", label="Avg Confidence")
        ax2.set_ylabel("Confidence")
        ax2.set_ylim(0, 1)

        for i, (cnt, conf) in enumerate(zip(counts, confidences)):
            ax1.text(i, cnt + 0.1, str(cnt), ha="center", va="bottom", fontsize=9)
            ax2.text(i, conf + 0.02, f"{conf:.2f}",
                     ha="center", va="bottom", fontsize=9)
            
        plt.title(f"{wafer_type} Detection Summary ({timestamp})")
        fig.tight_layout()
        png_filename = f"{wafer_type}_{timestamp}_detection_summary.png"
        plt.savefig(png_filename)
        plt.close()

    def save_log_txt(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"log_{timestamp}.txt"
            with open(log_filename, "w", encoding="utf-8") as f:
                f.write(self.text_edit.toPlainText())

            for wafer_type, stats in detection_stats.items():
                LogWindow.save_detection_summary_chart(wafer_type, stats, timestamp)

            QMessageBox.information(self, "Saved",
                                    f"Log({log_filename}) 및 detection summary 저장 완료.")
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"저장 실패:\n{e}")

class GraphCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(4, 2), dpi=100)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.x_data = list(range(50))
        self.y_data = [0] * 50
        self.ax.set_ylim(0, 100)
        self.line, = self.ax.plot(self.x_data, self.y_data, lw=2)
        self.ax.set_title("ALIGN ACCURACY(%)")
        self.ax.set_xlabel("FPS(TIME)")
        self.ax.set_ylabel("ACCURACY")

    def update_plot(self, value):
        self.y_data.pop(0)
        self.y_data.append(value)
        self.line.set_ydata(self.y_data)
        self.draw()

class VariableTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.default_values = {
            "HOME_DXDY": "(400, 200)",
            "Target_Point": "[640, 360]",
            "PIXEL_TO_MM_X": "0.6",
            "PIXEL_TO_MM_Y": "0.83",
            "TOLERANCE_PX": "1",
            "TOLERANCE_R": "1",
            "STEPS_PER_MM": "55",
            "MAX_STEPS": "32000",
        }
        self.fields = {}
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        for key, value in self.default_values.items():
            label = QLabel(key)
            line_edit = QLineEdit()
            line_edit.setText(value)
            self.fields[key] = line_edit
            form_layout.addRow(label, line_edit)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_changes)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_fields)

        layout.addLayout(form_layout)
        layout.addWidget(apply_button)
        layout.addWidget(reset_button)

        scroll_area = QScrollArea()
        container = QWidget()
        container.setLayout(layout)
        scroll_area.setWidget(container)
        scroll_area.setWidgetResizable(True)

        scroll_layout = QVBoxLayout()
        scroll_layout.addWidget(scroll_area)
        self.setLayout(scroll_layout)

    def apply_changes(self):
        try:
            changes = [] # 변수변경내역저장

            def update_variable(name, parse_func):
                old_val = str(globals()[name])
                new_val = parse_func(self.fields[name].text())
                if str(new_val) != old_val:
                    globals()[name] = new_val
                    changes.append(f"{name} changed from {old_val} to {new_val}.")
            update_variable("HOME_DXDY", eval)
            update_variable("Target_Point", eval)
            update_variable("PIXEL_TO_MM_X", float)
            update_variable("PIXEL_TO_MM_Y", float)
            update_variable("TOLERANCE_PX", int)
            update_variable("TOLERANCE_R", int,)
            update_variable("STEPS_PER_MM", int)
            update_variable("MAX_STEPS", int)

            for msg in changes:
                log(f"[CHANGED] {msg}", self.main_window)
        except Exception as e:
            print(f"[ERROR] Failed to apply changes: {e}")

    def reset_fields(self):
        for key, default_val in self.default_values.items():
            self.fields[key].setText(default_val)
        self.apply_changes()

class ZoomWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Alignment_Zoom")
        self.setFixedSize(600, 400)
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
      
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LOG")
        self.setGeometry(200, 200, 600, 400)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

        self.save_button = QPushButton("Save Log")
        self.save_button.clicked.connect(self.save_log_txt)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

    def append_log(self, msg):
        timestamp = time.strftime("[%Y-%m-%d]", time.localtime())
        self.text_edit.append(timestamp + msg)
        self.text_edit.verticalScrollBar().setValue(self.text_edit.verticalScrollBar().maximum())

    def save_log(self):
        try:
            now = time.strftime("%Y%m%d_%H%M%S")
            filename = f"log_{now}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.text_edit.toPlainText())
            QMessageBox.information(self, "Log Saved", f"Saved as {filename}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save log: {e}")

    def save_log_txt(self):
        try:
            with open("log.txt", "w", encoding="utf-8") as f:
                f.write(self.text_edit.toPlainText())
            for wafer_type, stats in detection_stats.items():
                LogWindow.save_detection_summary_chart(wafer_type, stats)
            QMessageBox.information(self, "Saved", "Log and detection summary saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save log and detection summary:\n{e}")

def main():
    app = QApplication(sys.argv)
    splash_pix = QPixmap("C:/Users/Mai/Desktop/wafer_project/splash.png").scaled( # SPLASH 이미지 주소 확인
        1400, 576, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )
    splash = QSplashScreen(splash_pix)
    splash.show()
    app.processEvents() 
    def start_main_program():
        splash.close()
        loop = QEventLoop(app)
        asyncio.set_event_loop(loop)
        window = MainWindow(loop)
        window.serial_ready = True
        with loop:
            loop.run_forever()
    QTimer.singleShot(3000, start_main_program)
    app.exec_()
if __name__ == "__main__":
    main()
