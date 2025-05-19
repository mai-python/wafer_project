import sys, os, gc, time, asyncio, cv2, torch, numpy as np, serial_asyncio 
from ultralytics import YOLO
from qasync import QEventLoop
from PyQt5.QtCore import Qt, QTimer 
from PyQt5.QtGui import QImage, QPixmap 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit, QInputDialog, QGroupBox,
    QVBoxLayout, QHBoxLayout, QFormLayout, QScrollArea, QComboBox, QSizePolicy, QLineEdit, QMessageBox, QTabWidget,
    QAction, QMenu, QSplashScreen, QGridLayout, QProgressBar, QFileDialog, QCheckBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
import matplotlib.pyplot as plt
import csv, random

model = YOLO("C:\\Users\\Mai\\Desktop\\wafer_project\\redline.pt") # best.pt check
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
Target_Point = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]
HOME_DXDY = (556, 188)
TOLERANCE_PX, TOLERANCE_R = 1, 1
PIXEL_TO_STEP = {'x': 33.0, 'y': 45.65}
MAX_STEPS = 32000
CAMERA_INDEX = 0
BAUDRATE = 9600
SERIAL_PORT = "COM5" #'Arduino IDE'들어가서 PORT 확인 필수
STABLE_REQUIRED = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE, FONT_THICKNESS = 0.6, 2
TEXT_COLOR = (255, 255, 255)
Target_degree, angle = 0, 0
DEGREE_TO_STEP, MAX_ROTATION_STEP = 1, 32000
random_x_min, random_x_max = 400, 800
random_y_min, random_y_max = 300, 500
random_angle_min, random_angle_max = 0, 360 
auto_loop_count = 0
CLASS_COLORS = {
    'F1': (0, 0, 50), 'F2': (0, 0, 100),
    'P100': (0, 0, 255), 'P111': (255, 0, 0),
}
detection_stats = {
    "P100": {"F1": [], "F2": [], "P100": [], "P111": []},
    "P111": {"F1": [], "F2": [], "P100": [], "P111": []}
}
confirmed_center = None
send_enabled = sending_command = awaiting_done = default_mode = False
stable_count = 0
last_sent_command = ""
setting_target_mode = force_send = auto_mode = False
wafer_type_override = None
has_f1 = has_f2 = False
fixed_f1_box = fixed_f2_box = None
corrected_box = []
position_done = False
rotation_done = False
RANDOM_MODE = False
position_accuracy_history = []
rotation_accuracy_history = []

def calculate_rotation_accuracy(angle_error_val, max_error=30):
    if angle_error_val <= 0.1:
        return 100.0
    elif angle_error_val >= max_error:
        return 0.0
    else:
        return round((1 - (angle_error_val / max_error)) * 100, 2)

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
    PERFECT_THRESH_PX = 3.0   

    if error_distance <= PERFECT_THRESH_PX: 
        return 100.0
    elif error_distance >= MAX_ERROR_PX:
        return 0.0 
    else:
        return round((1 - (error_distance / MAX_ERROR_PX)) * 100, 2)

def log(msg, window):
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    full_msg = f"{timestamp} {msg}" 
    #print(full_msg) # 터미널에 로그 출력
    if hasattr(window, 'log_window') and window.log_window:
        window.log_window.append_log(full_msg)

def compute_roundness(cnt):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0
    return 4 * np.pi * area / (perimeter * perimeter)


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
    raw = np.degrees(np.arctan2(dy, dx))
    if raw < 0:
        raw += 360
    return (180 - raw) % 360

def calculate_rotation(current_angle, target_angle):
    dR = (target_angle - current_angle + 360) % 360
    diR = 1
    stR = max(1, min(int(dR * DEGREE_TO_STEP), MAX_ROTATION_STEP))
    return diR, stR

def calculate_steps(from_point, to_point):
    dx = to_point[0] - from_point[0]
    dy = from_point[1] - to_point[1] 

    dx_steps = min(int(abs(dx) * PIXEL_TO_STEP['x']), MAX_STEPS)
    dy_steps = min(int(abs(dy) * PIXEL_TO_STEP['y']), MAX_STEPS)

    dix = 1 if dx > 0 else 0
    diy = 1 if dy > 0 else 0
    command = f"{dix},{diy},{dx_steps},{dy_steps},0,0"
    return dx, dy, dx_steps, dy_steps, command

class MainWindow(QMainWindow):
    def __init__(self, loop): 
        super().__init__() 
        self.loop = loop
        self.mode = "IDLE"
        self.setWindowTitle("Wafer_Aligner")
        self.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT + 200 + self.menuBar().height())
        self.setMouseTracking(True)
        menubar = self.menuBar()

        self.recording = False
        self.video_writer = None

        self.log_time = 0

        self.label_wafer_type      = QLabel()
        self.label_state_label     = QLabel()
        self.label_target_point    = QLabel()
        self.label_coords          = QLabel()
        self.label_target_degree   = QLabel()
        self.label_angle_error     = QLabel()

        self.language = "en"
        self.texts = {
            "ko": {
                "align": "자동모드",
                "Align": "자동모드",
                "ALGIN": "자동모드",
                "set target": "목표설정",
                "Set_Target": "목표설정",
                "SET_TARGET": "목표설정",
                "SET TARGET": "목표설정",
                "set_target": "목표설정",
                "DEFAULT": "원위치",
                "DEFAULT": "원위치",
                "default": "원위치",
                "Default": "원위치",
                "Angle": "각도",
                "ANGLE": "각도",
                "angle": "각도",
                "ROTATION": "회전",
                "Rotation": "회전",
                "rotation": "회전",
                "Status": "상태",
                "STATUS": "상태",
                "status": "상태",
                "Standby": "대기중",
                "STANDBY": "대기중",
                "standby": "대기중",
                "START": "시작",
                "Start": "시작",
                "start": "시작",
                "help": "도움말",
                "Help": "도움말",
                "file": "파일",
                "File": "파일",
                "Capture": "사진저장",
                "capture": "사진저장",
                "Exit": "종료",
                "exit": "종료",
                "position": "위치이동",
                "POSITION": "위치이동",
                "Position": "위치이동",
                "Variables": "변수설정",
                "variables": "변수설정",
                "Graph": "그래프",
                "graph": "그래프",
                "Reload": "모델 새로고침",
                "reload": "모델 새로고침",
                "Mode": "모드",
                "Log": "로그",
                "log": "로그",
                "Tool": "도구",
                "Language": "언어",
                "language": "언어",
                "Wafer Type": "웨이퍼 종류",
                "Position_Err": "좌표_오차",
                "Angle_Err": "각도_오차",
                "Positioning...": "위치이동중...",
                "Defaulting...": "원위치중",
                "Rotationing...": "회전중...",
                "Click to set Target": "목표지점 설정하세요",
                "POSITION completed": "이동 완료됨",
                "Waiting for next Wafer": "다음 웨이퍼 기다리는중",
                "Function": "기능",
                "Settings": "설정",
                "Wafer Type": "웨이퍼 종류",
                "Position_Err": "좌표 오차",
                "Target Point": "타겟 좌표",
                "Target Deg.": "목표 각도",
                "Angle_Err": "각도 오차",
                "Status": "상태",
                "ALIGN ACCURACY(%)": "정렬 정확도(%)",
                "ROTATION ACCURACY": "회전 정확도",
                "Target Point": "타겟 좌표",
                "Target Deg.": "목표 각도",
                "Saved as": "저장 위치:",
                "Screenshot": "스크린샷",
                "Apply": "적용",
                "Reset": "초기화",
                "Alignment_Zoom": "확대창",
                "Log Saved": "로그 저장됨",
                "Saved": "저장됨",
                "Error": "오류",
                "Do you want to save the log and summary before exiting?": "종료 전에 로그와 요약을 저장하시겠습니까?",
                "Save Error": "저장 오류",
                "Wafer Type": "웨이퍼 종류",
                "Target Point": "타겟 좌표",
                "Position_Err": "좌표_오차",
                "Target Deg.": "목표 각도",
                "Angle_Err": "각도_오차",
                "Standby": "대기중",
                "standby": "대기중",
                "STANDBY": "대기중",
                "Positioning...": "위치이동중...",
                "Defaulting...": "원위치중...",
                "Rotationing...": "회전중...",
                "Click to set Target": "목표지점 설정하세요",
                "POSITION completed": "이동 완료됨",
                "Waiting for next Wafer": "다음 웨이퍼 기다리는중",
                "Align...": "정렬중...",
                "ALIGNING": "정렬중",
                "ROTATING": "회전중",
                "DEFAULT": "원위치중",
                "AUTO": "자동모드",
                "MANUAL": "수동모드",
                "Operation Not Allowed": "작업 불가",
                "Recording": "녹화",
                "Recording started": "녹화 시작됨",
                "Recording completed": "녹화 완료됨"
            },
            "en": {
                "align": "Align",
                "Set Target": "Set Target",
                "DEFAULT": "DEFAULT",
                "angle": "Angle",
                "rotation": "Rotation",
                "auto": "Auto Mode",
                "status": "Status",
                "standby": "Standby",
                "start": "Start",
                "set_target": "Set Target",
                "help": "Help",
                "file": "File",
                "capture": "Capture",
                "exit": "Exit",
                "position": "Position",
                "variables": "Variables",
                "graph": "Graph",
                "reload": "Reload Model",
                "log": "Log",
                "Status": "Status",
                "STANDBY": "STANDBY",
                "ALIGNING": "ALIGNING",
                "ROTATING": "ROTATING",
                "DEFAULT": "DEFAULT",
                "AUTO": "AUTO",
                "MANUAL": "MANUAL",
                "Standby": "Standby",
                "Positioning...": "Positioning...",
                "Rotationing...": "Rotationing...",
                "DEFAULT...": "DEFAULT...",
                "Align...": "Align...",
                "Click to set Target": "Click to set Target",
                "Function": "Function",
                "Settings": "Settings",
                "ALIGN ACCURACY(%)": "ALIGN ACCURACY(%)",
                "ROTATION ACCURACY": "ROTATION ACCURACY",
                "Target Point": "Target Point",
                "Target Deg.": "Target Deg.",
                "Saved as": "Saved as",
                "Screenshot": "Screenshot",
                "Apply": "Apply",
                "Reset": "Reset",
                "Alignment_Zoom": "Alignment_Zoom",
                "Log Saved": "Log Saved",
                "Saved": "Saved",
                "Error": "Error",
                "Do you want to save the log and summary before exiting?": "Do you want to save the log and summary before exiting?",
                "Save Error": "Save Error",
                "Recording": "Recording",
                "Recording started": "Recording started",
                "Recording completed": "Recording completed"
            }
        }
        self.loop_results = []  # [ (loop_idx, pos_acc, rot_acc), ... ]

        self.file_menu = menubar.addMenu(self.t("File"))
        self.capture_action = QAction(self.t("Capture"), self)
        self.save_all_action = QAction("Save", self, triggered=self.save_all)
        self.exit_action = QAction(self.t("Exit"), self, triggered=self.close)
        self.file_menu.addAction(self.capture_action)
        self.file_menu.addAction(self.save_all_action)
        self.file_menu.addAction(self.exit_action)
        self.capture_action.triggered.connect(self.capture_screenshot)

        self.mode_menu = menubar.addMenu("Mode")
        self.auto_mode_action = QAction("Auto Mode", self, triggered=self.set_auto_mode)
        self.manual_mode_action = QAction("Manual Mode", self, triggered=self.set_manual_mode)
        self.idle_mode_action = QAction("Idle Mode", self, triggered=self.set_idle_mode)  # 추가

        self.mode_menu.addAction(self.auto_mode_action)
        self.mode_menu.addAction(self.manual_mode_action)
        self.mode_menu.addAction(self.idle_mode_action)  
        self.update_state("IDLE")
        self.idle_timer = QTimer()
        self.idle_timer.timeout.connect(self.enter_idle_mode)

        self.tool_menu = menubar.addMenu(self.t("Tool"))
        self.reload_action = QAction(self.t("Reload"), self, triggered=self.reload_yolo_model)
        self.log_action = QAction(self.t("Log"), self, triggered=self.toggle_log_window)
        self.record_action = QAction(self.t("Recording"))
        self.record_action.setCheckable(True)
        self.record_action.triggered.connect(self.toggle_recording)
        self.tool_menu.addAction(self.record_action)
        self.tool_menu.addAction(self.reload_action)
        self.tool_menu.addAction(self.log_action)
    
        self.settings_menu = menubar.addMenu(self.t("Settings"))
        self.variables_action = QAction(self.t("Variables"), self, triggered=self.show_variable_tab)
        self.settings_menu.addAction(self.variables_action)
        self.language_menu = QMenu(self.t("Language"), self)
        self.ko_action = QAction("한국어", self)
        self.en_action = QAction("English", self)
        self.language_menu.addAction(self.ko_action)
        self.language_menu.addAction(self.en_action)
        self.settings_menu.addMenu(self.language_menu)
        self.ko_action.triggered.connect(lambda: self.set_language("ko"))
        self.en_action.triggered.connect(lambda: self.set_language("en"))
        
        self.help_action = QAction(self.t("Help"), self)
        self.help_action.triggered.connect(self.show_help_dialog)
        menubar.addAction(self.help_action)
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(3, FRAME_WIDTH) 
        self.cap.set(4, FRAME_HEIGHT)

        self.summary_type_inited = {"P100": False, "P111": False}
        self.zoom_window = None
        self.log_window = LogWindow() 
        self.rotation_active = False 
        self.warned_once = False 

        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.hide() 
        self.setup_tabs() 
        self.align_label = QLabel(self.t("ALIGN ACCURACY(%)"))
        self.align_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.rotation_label = QLabel(self.t("ROTATION ACCURACY"))
        self.rotation_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.align_bar = QProgressBar()
        self.rotation_bar = QProgressBar()
        self.align_bar.setMaximum(100)
        self.rotation_bar.setMaximum(100)
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(self.align_label)
        progress_layout.addWidget(self.align_bar)
        progress_layout.addWidget(self.rotation_label)
        progress_layout.addWidget(self.rotation_bar)
        self.progress_widget = QWidget()
        self.progress_widget.setLayout(progress_layout)
        self.progress_widget.setFixedWidth(300)
        self.label_wafer_type.setText(self.t("Wafer Type") + ": -")
        self.label_target_point.setText(self.t("Target Point:") + f"\n({Target_Point[0]}, {Target_Point[1]})")
        self.label_coords.setText(self.t("Position_Err") + ":\n" + "(0, 0)")
        self.label_target_degree.setText(self.t("Target Deg.") + f": {Target_degree}°")
        self.label_angle_error.setText(self.t("Angle_Err") + f": 0°")
        for lbl in [
        self.label_wafer_type,
        self.label_state_label,
        self.label_target_point,
        self.label_coords,
        self.label_target_degree,
        self.label_angle_error
        ]:
            lbl.setStyleSheet("font-size: 15px; padding: 10px;")

        status_layout = QGridLayout()
        status_layout.addWidget(self.label_wafer_type, 0, 0)
        status_layout.addWidget(self.label_state_label, 1, 0)
        status_layout.addWidget(self.label_target_point, 0, 1)
        status_layout.addWidget(self.label_coords, 0, 2)
        status_layout.addWidget(self.label_target_degree, 1, 1)
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
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.setMouseTracking(True)
        self.show()
        asyncio.create_task(self.serial_setup())
        self.state = "Idle"

    def log_time_limit(self, msg):
        now = time.time()
        if now - self.log_time >= 10:
            self.log_time = now
            log(msg, self)

    def save_all(self):
        fn = QFileDialog.getSaveFileName(self, "Save Loop Results", "", "CSV Files (*.csv)")[0]
        if not fn:
            return
        with open(fn, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Loop", "Position Acc (%)", "ΔX (px)", "ΔY (px)",
                "Rotation Acc (%)", "Angle Err (°)",
                "P100_F1_Count", "P100_F2_Count", "P100_P100_Count", "P100_P111_Count",
                "P111_F1_Count", "P111_F2_Count", "P111_P100_Count", "P111_P111_Count"
            ])
            for row in self.loop_results:
                writer.writerow(row)
        QMessageBox.information(self, "Saved", f"Loop results saved to {fn}")

    def set_idle_mode(self):
        self.mode = "IDLE"
        self.tabs.hide()
        self.update_state("IDLE")
        log("[MODE] Entered IDLE mode (수동 진입)", self)

    def set_auto_mode(self):
        self.mode = "AUTO"
        self.tabs.clear()
        self.tabs.addTab(self.auto_tab, self.t("Align")) 
        self.update_state("AUTO")
        self.tabs.show()
        self.reset_idle_timer()

    def set_manual_mode(self):
        self.mode = "MANUAL"
        self.update_state("MANUAL")
        self.tabs.clear()
        self.tabs.addTab(self.align_tab, self.t("Position"))
        self.tabs.addTab(self.angle_tab, self.t("Rotation"))
        self.tabs.show()
        self.reset_idle_timer()

    def enter_idle_mode(self):
        self.mode = "IDLE"
        self.update_state("IDLE")
        self.tabs.hide()
        log("[MODE] Entered IDLE mode", self)

    def reset_idle_timer(self):
        self.idle_timer.start(120000)  # 1분

    def update_state(self, mode):
        if mode.upper() == "AUTO":
            self.label_state_label.setText(self.t("Auto"))
        elif mode.upper() == "MANUAL":
            self.label_state_label.setText(self.t("Manual"))
        elif mode.upper() == "IDLE":
            self.label_state_label.setText(self.t("Idle"))

    def toggle_recording(self, checked):
        if checked:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"record_{timestamp}.avi"
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT)
            )
            self.recording = True
            log("[RECORDING] Record started", self)
        else:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            log("[RECORDING] Record completed")

    def set_language(self, lang):
        self.language = lang
        self.update_all_texts()

    def t(self, key):
        return self.texts[self.language].get(key, key)

    def update_all_texts(self):
        try:
            # 메뉴
            self.file_menu.setTitle(self.t("File"))
            self.capture_action.setText(self.t("Capture"))
            self.save_all_action.setText(self.t("Save"))

            self.exit_action.setText(self.t("Exit"))
            self.auto_mode_action.setText(self.t("Align"))
            self.manual_mode_action.setText(self.t("Position"))
            self.manual_mode_action.setText(self.t("Rotation"))
            self.tool_menu.setTitle(self.t("Tool"))
            self.reload_action.setText(self.t("Reload"))
            self.log_action.setText(self.t("Log"))
            self.settings_menu.setTitle(self.t("Settings"))
            self.variables_action.setText(self.t("Variables"))
            self.language_menu.setTitle(self.t("Language"))
            self.ko_action.setText("한국어")
            self.en_action.setText("English")
            self.help_action.setText(self.t("Help"))
            # 탭
            self.tabs.setTabText(self.tabs.indexOf(self.align_tab), self.t("Position"))
            self.tabs.setTabText(self.tabs.indexOf(self.angle_tab), self.t("Rotation"))
            self.tabs.setTabText(self.tabs.indexOf(self.variable_tab), self.t("Variables"))
            self.tabs.setTabText(self.tabs.indexOf(self.auto_tab), self.t("Align"))
            # 버튼/라벨
            self.m_button.setText(self.t("SET TARGET"))
            self.s_button.setText(self.t("START"))
            self.u_button.setText(self.t("DEFAULT"))
            self.a_button.setText(self.t("SET TARGET"))
            self.r_button.setText(self.t("START"))
            self.t_button.setText(self.t("Align"))
            self.align_label.setText(self.t("ALIGN ACCURACY(%)"))
            self.rotation_label.setText(self.t("ROTATION ACCURACY(%)"))
            self.label_wafer_type.setText(self.t("Wafer Type") + ": -")
            self.label_state_label.setText(self.t("Mode:"))
            self.label_coords.setText(self.t("Position_Err") + ":\n" + "(0, 0)")
            self.label_target_point.setText(self.t("Target Point") + f":\n({Target_Point[0]}, {Target_Point[1]})")
            self.label_target_degree.setText(self.t("Target Deg.") + f":{Target_degree}°")
            self.label_angle_error.setText(self.t("Angle_Err") + f":0°")
        except Exception as e:
            print("update_all_texts 오류:", e)

    def update_target_point_label(self):
        self.label_target_point.setText(self.t("Target Point") + f":\n({Target_Point[0]}, {Target_Point[1]})")

    def apply_auto_var_changes(self):
        try:
            global HOME_DXDY, Target_Point, Target_degree, auto_loop_count, RANDOM_MODE, RANDOM_DEG_MAX, RANDOM_DEG_MIN, RANDOM_X_MAX, RANDOM_X_MIN, RANDOM_Y_MAX, RANDOM_Y_MIN
            HOME_DXDY     = eval(self.auto_var_fields["HOME_DXDY"].text())
            Target_Point  = eval(self.auto_var_fields["TARGET_POINT"].text())
            Target_degree = float(self.auto_var_fields["TARGET_DEGREE"].text())
            auto_loop_count = int(self.auto_var_fields["LOOP"].text())
            RANDOM_X_MIN      = int(self.auto_var_fields["RANDOM_X_MIN"].text())
            RANDOM_X_MAX      = int(self.auto_var_fields["RANDOM_X_MAX"].text())
            RANDOM_Y_MIN      = int(self.auto_var_fields["RANDOM_Y_MIN"].text())
            RANDOM_Y_MAX      = int(self.auto_var_fields["RANDOM_Y_MAX"].text())
            RANDOM_DEG_MIN    = float(self.auto_var_fields["RANDOM_DEG_MIN"].text())
            RANDOM_DEG_MAX    = float(self.auto_var_fields["RANDOM_DEG_MAX"].text())

            RANDOM_MODE = self.random_checkbox.isChecked()

            self.update_target_point_label()
            self.label_target_degree.setText(self.t("Target Deg.") + f":{Target_degree}°")
        except Exception as e:
            show_warning(self, "Apply Error", str(e))

    def reset_auto_var_fields(self):
        self.auto_var_fields["HOME_DXDY"].setText(str((556, 188)))
        self.auto_var_fields["TARGET_POINT"].setText(str([FRAME_WIDTH // 2, FRAME_HEIGHT // 2]))
        self.auto_var_fields["TARGET_DEGREE"].setText(str(0))
        # 방금 수정한 함수 호출
        self.apply_auto_var_changes()

    def setup_tabs(self):
        # Align 탭
        self.align_tab = QWidget()
        align_layout = QVBoxLayout()
        self.m_button = QPushButton(self.t("SET TARGET"))
        self.s_button = QPushButton(self.t("START"))
        self.u_button = QPushButton(self.t("DEFAULT"))
        for btn in [self.m_button, self.s_button, self.u_button]:
            btn.setFixedHeight(50)
            btn.setFixedWidth(460)
            btn.setStyleSheet("font-size: 20px; padding: 6px;")
            align_layout.addWidget(btn)
        self.align_tab.setLayout(align_layout)
        self.s_button.clicked.connect(self.toggle_send)
        self.m_button.clicked.connect(self.toggle_target_mode)
        self.u_button.clicked.connect(self.default)
        # Angle(회전) 탭
        self.angle_tab = QWidget()
        angle_layout = QVBoxLayout()
        self.a_button = QPushButton(self.t("SET TARGET"))
        self.r_button = QPushButton(self.t("START"))
        for btn in [self.a_button, self.r_button]:
            btn.setFixedHeight(70)
            btn.setFixedWidth(460)
            btn.setStyleSheet("font-size: 20px; padding: 6px;")
            angle_layout.addWidget(btn)
        self.angle_tab.setLayout(angle_layout)
        self.a_button.clicked.connect(self.set_target_degree)
        self.r_button.clicked.connect(lambda: asyncio.create_task(self.start_rotation()))
        # Auto 탭
        self.auto_tab = QWidget()
        adv_layout = QVBoxLayout()
        self.t_button = QPushButton(self.t("ALIGN"))    
        self.t_button.setFixedHeight(30)
        self.t_button.setFixedWidth(500)
        self.t_button.setStyleSheet("font-size: 20px; padding: 6px;")
        adv_layout.addWidget(self.t_button)
        self.t_button.clicked.connect(self.start_auto_align)

        self.random_checkbox = QCheckBox("RANDOM MODE")
        adv_layout.addWidget(self.random_checkbox)

        self.auto_var_fields = {}
        default_values = {
            "HOME_DXDY": str(HOME_DXDY),
            "TARGET_POINT": str(Target_Point),
            "TARGET_DEGREE": str(Target_degree),
            "LOOP": str(auto_loop_count),
            "RANDOM_X_MIN": str(random_x_min),
            "RANDOM_X_MAX": str(random_x_max),
            "RANDOM_Y_MIN": str(random_y_min),
            "RANDOM_Y_MAX": str(random_y_max),
            "RANDOM_ANGLE_MIN": str(random_angle_min),
            "RANDOM_ANGLE_MAX": str(random_angle_max),
        }
        form_layout = QFormLayout()
        for key, val in default_values.items():
            label = QLabel(key)
            line_edit = QLineEdit(val)
            self.auto_var_fields[key] = line_edit
            form_layout.addRow(label, line_edit)

        apply_btn = QPushButton(self.t("Apply"))
        reset_btn = QPushButton(self.t("Reset"))
        apply_btn.clicked.connect(self.apply_auto_var_changes)
        reset_btn.clicked.connect(self.reset_auto_var_fields)
        btn_box = QHBoxLayout()
        btn_box.addWidget(apply_btn)
        btn_box.addWidget(reset_btn)

        form_widget = QWidget()
        form_widget.setLayout(form_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)

        adv_layout.addWidget(scroll)
        adv_layout.addLayout(btn_box)

        self.auto_tab.setLayout(adv_layout)
        # Variable 탭
        self.variable_tab = VariableTab(self)

    def start_auto_align(self):
        global auto_mode, rotation_done, position_done
        if send_enabled or self.rotation_active or default_mode:
            log("[ERROR] Cannot start AUTO now", self)
            self.show_warning(self.t("Operation Not Allowed", "AUTO cannot be excuted at this time"))
            return
        self.update_state(self.t("AUTO"))
        rotation_done = position_done = False
        self.reset_progress()
        auto_mode = True
        if self.loop and not self.loop.is_closed():
            log("[AUTO] Started", self)
            self.loop.create_task(self.auto_align_loop())
        else:
            log("[ERROR] Event loop is not ready or already closed", self)

    def show_align_tab(self):
        if self.tabs.indexOf(self.align_tab) == -1:
            self.tabs.addTab(self.align_tab, self.t("Position"))
        self.tabs.show()
        self.tabs.setCurrentWidget(self.align_tab)

    def show_angle_tab(self):
        global position_done, rotation_done
        position_done = rotation_done = False
        if self.tabs.indexOf(self.angle_tab) == -1:
            self.tabs.addTab(self.angle_tab, self.t("Rotation"))
        self.tabs.show()
        self.tabs.setCurrentWidget(self.angle_tab)

    def show_auto_tab(self):
        if self.tabs.indexOf(self.auto_tab) == -1:
            self.tabs.addTab(self.auto_tab, self.t("Align"))
        self.tabs.show()
        self.tabs.setCurrentWidget(self.auto_tab)

    def update_progress(self, dx, dy, angle_error, mode):
        if mode == "align":
            if position_done:
                self.align_bar.setValue(100)
                return

            ox, oy = self.initial_offset

            start_distance = (ox ** 2 + oy ** 2) ** 0.5
            current_distance = (dx ** 2 + dy ** 2) ** 0.5

            if start_distance == 0:
                prog = 100
            else:
                prog = max(0, min(1, 1 - current_distance / start_distance)) * 100

            self.align_bar.setValue(int(prog))
    def reset_progress(self):
        self.align_bar.setValue(0)
        self.rotation_bar.setValue(0)

    def set_target_degree(self):
        if send_enabled or auto_mode or default_mode or self.rotation_active:
            log("[ERROR] Cannot set TARGET_DEGREE during active operation", self)
            show_warning(self,
                         "Operation Not Allowed",
                         "Cannot set target angle while another operation is in progress.")
            return

        degree, ok = QInputDialog.getDouble(
            self,
            "SETTING",
            "INPUT DEGREE",
            decimals=1,
            min=0,
            max=360
        )
        if not ok:
            return

        self.reset_progress()
        global Target_degree
        Target_degree = degree
        log(f"[Angle] Set Target_degree: {Target_degree:.0f}°", self)
        self.label_target_degree.setText(f"Target Deg.: {Target_degree:.0f}°")

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Exit",
            "Do you want to save the log and summary before exiting?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        if reply == QMessageBox.Yes:
            try:
                self.log_window.save_log_txt()
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save logs:\n{e}")
            event.accept()
        elif reply == QMessageBox.No:
            event.accept()
        else:
            event.ignore()

    async def serial_setup(self):
        try:
            self.ser_reader, self.ser_writer = await serial_asyncio.open_serial_connection(
                url=SERIAL_PORT, baudrate=BAUDRATE)
            self.serial_ready = True
            log("[INIT] Serial ready", self)
        except Exception as e:
            log(f"[ERR] Serial setup failed: {e}", self)

    async def default_loop(self): # 개선완료
        global confirmed_center, default_mode, dx, dy, send_enabled
        self.reset_progress()
        if not confirmed_center:
            return

        default_mode = True
        log("[DEFAULT] DEFAULT started", self)

        MAX_RETRY = 60  
        retry = 0
        while retry < MAX_RETRY:
            await asyncio.sleep(0.5) 
            ret, frame = self.cap.read()
            if not ret:
                continue
            results = model(frame)
            wafer_center = None
            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                label = results[0].names[int(cls)]
                if label in ["P100", "P111"]:
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    wafer_center = (cx, cy)
            if wafer_center:
                confirmed_center = wafer_center

            dx, dy, dx_steps, dy_steps, command = calculate_steps(confirmed_center, HOME_DXDY)
            if abs(dx) + abs(dy) <= 1:
                log(f"[DEFAULT] DEFAULT completed (dx={dx}, dy={dy})", self)
                break

            await self.send_serial_command(command)
            await asyncio.sleep(0.7)  
            retry += 1
        else:
            log("[DEFAULT] DEFAULT Timeout...", self)
        default_mode = False
        await asyncio.sleep(4)
        log("[AUTO] AUTO_Standby ", self)

    async def send_serial_command(self, command):
        global sending_command, awaiting_done, last_sent_command, force_send, default_mode, send_enabled

        if not self.serial_ready or self.ser_writer is None:
            return

        critical_stop = command.upper() in ("STOP", "STOP_ALIGNMENT")
        if (sending_command or awaiting_done) and not critical_stop:
            return
        if command == last_sent_command and not force_send and not critical_stop:
            return

        sending_command = True
        awaiting_done = True

        if default_mode:
            log("[DEFAULT] DEFAULT")
        elif command == "STOP_ALIGNMENT":
            log("[ALIGN] ALIGN stopped", self)
        else:
            actual = command
            if command.upper() == "STOP" or command == "STOP_ALIGNMENT":
                actual = "0,0,0,0,0,0"

            parts = actual.split(",")
            if len(parts) == 6 and (int(parts[4]) != 0 or int(parts[5]) != 0):
                log(f"[ROTATION] Sending ROTATION command: {actual}", self)
            else:
                log(f"[ALIGN] Sending ALIGN command: {actual}", self)

            command = actual

        try:
            self.ser_writer.write((command + "\n").encode())
            await self.ser_writer.drain()
            last_sent_command = command
            force_send = False

            while True:
                try:
                    line = await asyncio.wait_for(self.ser_reader.readline(), timeout=30.0)
                    decoded = line.decode().strip()
                    if "DONE" in decoded or "READY" in decoded:
                        break
                except asyncio.TimeoutError:
                    if default_mode:
                        log("[DEFAULT] TIMEOUT", self)
                    elif command == "STOP_ALIGNMENT":
                        log("[ALIGN] TIMEOUT ", self)
                    break

        except Exception as e:
            log(f"[ERR] Serial communication error: {e}", self)
            log("[ARDUINO] SERIAL_ERROR", self)

        finally:
            sending_command = False
            awaiting_done = False

            if default_mode:
                log("[DEFAULT] DONE", self)
                default_mode = False
                send_enabled = False

            elif last_sent_command == "0,0,0,0,0,0":
                log("[STATUS] STANDBY")
            else:
                parts = last_sent_command.split(",")
                if len(parts) == 6 and (int(parts[4]) != 0 or int(parts[5]) != 0):
                    log("[ROTATION] STEP_DONE", self)
                else:
                    log("[ALIGN] STEP DONE")

    def toggle_log_window(self):
        if self.log_window.isVisible():
            self.log_window.hide()
        else:
            self.log_window.show()
            self.log_window.raise_()

    def toggle_send(self):  # ALIGN 함수
        global send_enabled, stable_count, Target_Point, auto_mode, position_done, rotation_done

        if self.rotation_active or auto_mode or default_mode:
            log("[ERROR] Cannot start ALIGN during other operation", self)
            self.show_warning("Operation Not Allowed", "ALIGN cannot be excuted at this time")
            return # 로그 출력

        if send_enabled:
            send_enabled = False
            self.reset_progress()
            asyncio.create_task(self.send_serial_command("STOP_ALIGNMENT"))
            log("[ALIGN] ALIGN stopped", self)
            return

        position_done = rotation_done = False
        self.reset_progress()
        send_enabled = True
        auto_mode = False

        log("[ALIGN] ALIGN started", self)
        stable_count = 0
        self.warned_once = False
        if confirmed_center:
            dx0 = abs(Target_Point[0] - confirmed_center[0])
            dy0 = abs(Target_Point[1] - confirmed_center[1])
            self.initial_offset = (dx0, dy0)
        else:
            self.initial_offset = (FRAME_WIDTH, FRAME_HEIGHT)

    async def start_rotation(self):
        global angle, rotation_done
        if send_enabled or auto_mode or default_mode or self.rotation_active:
            log("[ERROR] Cannot start ROTATION during other operation", self)
            self.show_warning("Operation Not Allowed", "Rotation cannot be executed at this time.")
            return

        start_ang = angle
        total = (Target_degree - start_ang) % 360
        if total == 0:
            total = 360
        self.rotation_start = start_ang
        self.rotation_total = total

        rotation_done = False
        self.rotation_active = True
        log("[ROTATION] Rotation started", self)

        while True:
            await asyncio.sleep(0.2)
            ret, frame = self.cap.read()
            if not ret:
                continue
            # ─── YOLO로 F1·웨이퍼 중심 검출 ───
            results = model(frame)
            f1_center = wafer_center = None
            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                label = results[0].names[int(cls)]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if label in ["P100", "P111"]:
                    wafer_center = (cx, cy)
                elif label == "F1":
                    f1_center = (cx, cy)
            if not f1_center or not wafer_center:
                log("[DEBUG] Can't find both F1 and wafer center → skipping rotation step", self)
                continue

            angle = calculate_angle(f1_center, wafer_center)
            diff  = (Target_degree - angle + 540) % 360 - 180
            angle_error = abs(diff)

            traveled = (angle - self.rotation_start + 360) % 360
            prog     = min(traveled / self.rotation_total, 1.0) * 100
            self.rotation_bar.setValue(int(prog))

            self.label_angle_error.setText(f"Angle_Err: {angle_error:.2f}°")
            # ⑤ 완료 판정
            if angle_error < TOLERANCE_R:
                rotation_done = True
                log("[ROTATION] Rotation completed", self)
                await self.send_serial_command("STOP")
                self.rotation_active = False
                self.update_points_after_rotation(frame)
                break
            else:
                # ⑥ 계속 회전 명령
                diR, stR = calculate_rotation(angle, Target_degree)
                command = f"0,0,0,0,{diR},{stR}"
                log(f"[ROTATION] Sending step command: {command}", self)
                await self.send_serial_command(command)

    def update_points_after_rotation(self, frame):
        global confirmed_center
        results = model(frame)
        wafer_center = None
        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            label = results[0].names[int(cls)]
            if label in ["P100", "P111"]:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                wafer_center = (cx, cy)
        if wafer_center:
            confirmed_center = wafer_center
            log(f"[ROTATION] Updated center: {confirmed_center}", self)

    def reload_yolo_model(self):
        global model
        log("[YOLO] reload model...", self)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = YOLO("redline.pt")
        log("[YOLO]Successfully reload model", self)

    async def auto_align_loop(self):
        global auto_mode, send_enabled, home_mode, force_send, confirmed_center
        global align_done, rotation_done, auto_loop_count, Target_Point, Target_degree

        log("[AUTO] AUTO started", self)
        auto_mode = True
        count = 0

        while auto_mode and (auto_loop_count == 0 or count < auto_loop_count):
            # ── 랜덤 모드 타겟 설정 ──
            if RANDOM_MODE:
                Target_Point = [
                    random.randint(random_x_min, random_x_max),
                    random.randint(random_y_min, random_y_max)
                ]
                Target_degree = random.uniform(random_angle_min, random_angle_max)
                self.update_target_point_label()
                self.label_target_degree.setText(
                    self.t("Target Deg.") + f": {Target_degree:.1f}°"
                )
                log(f"[AUTO] New random target → Point: {Target_Point}, Degree: {Target_degree:.1f}°", self)
                await asyncio.sleep(0.2)

            # 1) DEFAULT 단계
            if confirmed_center:
                home_mode = True
                log(f"[AUTO] Loop {count+1}: DEFAULT_1 started", self)
                stable_count_default = 0
                retry = 0
                while retry < 200:
                    dx, dy, dx_s, dy_s, cmd = calculate_steps(confirmed_center, HOME_DXDY)
                    if abs(dx) > TOLERANCE_PX or abs(dy) > TOLERANCE_PX:
                        await self.send_serial_command(cmd)
                        await asyncio.sleep(0.05)
                        retry += 1
                    else:
                        stable_count_default += 1
                        if stable_count_default >= STABLE_REQUIRED:
                            break
                        await asyncio.sleep(0.05)
                log(f"[AUTO] Loop {count+1}): DEFAULT_1 completed ", self)


            # 2) ALIGN 단계
            log(f"[AUTO] Loop {count+1}: [AUTO] ALIGN started", self)
            align_done = False
            stable_count = 0
            if confirmed_center:
                dx0 = abs(Target_Point[0] - confirmed_center[0])
                dy0 = abs(Target_Point[1] - confirmed_center[1])
                self.initial_offset = (dx0, dy0)
            else:
                self.initial_offset = (FRAME_WIDTH, FRAME_HEIGHT)

            position_accuracy_history.clear()
            retry = 0
            while retry < 200:
                await asyncio.sleep(0.05)
                if confirmed_center:
                    dx, dy, dx_s, dy_s, cmd = calculate_steps(confirmed_center, Target_Point)
                    pos_acc = calculate_accuracy_px(Target_Point, confirmed_center)
                    position_accuracy_history.append(pos_acc)
                    self.align_bar.setValue(int(pos_acc))
                    if abs(dx) > TOLERANCE_PX or abs(dy) > TOLERANCE_PX:
                        await self.send_serial_command(cmd)
                    else:
                        stable_count += 1
                        if stable_count >= STABLE_REQUIRED:
                            align_done = True
                            break
                retry += 1

            await self.send_serial_command("STOP_ALIGN")
            log(f"[AUTO] Loop {count+1}: ALIGN done", self)
            await asyncio.sleep(3.0)

            # 3) ROTATION 단계
            log(f"[AUTO] Loop {count+1}: ROTATION", self)
            rotation_done = False
            self.rotation_active = True
            final_rot_acc = 0.0
            final_angle_err = 0.0

            while True:
                await asyncio.sleep(0.2)
                ret, frame = self.cap.read()
                if not ret:
                    continue

                results = model(frame)
                f1_c = wafer_c = None
                for b in results[0].boxes.data:
                    x1, y1, x2, y2, *_ = b.tolist()
                    lbl = results[0].names[int(b[5])]
                    cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                    if lbl in ["P100", "P111"]:
                        wafer_c = (cx, cy)
                    elif lbl == "F1":
                        f1_c = (cx, cy)
                if not f1_c or not wafer_c:
                    continue

                cur_ang = calculate_angle(f1_c, wafer_c)
                diff = (Target_degree - cur_ang + 540) % 360 - 180
                angle_err = abs(diff)
                rot_acc = calculate_rotation_accuracy(angle_err)

                self.rotation_bar.setValue(int(rot_acc))

                if angle_err < TOLERANCE_R:
                    final_rot_acc = rot_acc
                    final_angle_err = angle_err
                    await self.send_serial_command("STOP")
                    self.rotation_active = False
                    self.update_points_after_rotation(frame)
                    rotation_done = True
                    break
                else:
                    diR, stR = calculate_rotation(angle, Target_degree)
                    cmd = f"0,0,0,0,{diR},{stR}"
                    await self.send_serial_command(cmd)

            # ↪ 결과 기록
            final_pos_acc = position_accuracy_history[-1] if position_accuracy_history else 0.0
            # 최종 X,Y 오차 계산
            final_dx = Target_Point[0] - confirmed_center[0] if confirmed_center else 0
            final_dy = Target_Point[1] - confirmed_center[1] if confirmed_center else 0

            counts = [
                len(detection_stats["P100"]["F1"]), len(detection_stats["P100"]["F2"]),
                len(detection_stats["P100"]["P100"]), len(detection_stats["P100"]["P111"]),
                len(detection_stats["P111"]["F1"]), len(detection_stats["P111"]["F2"]),
                len(detection_stats["P111"]["P100"]), len(detection_stats["P111"]["P111"])
            ]

            self.loop_results.append([
                count + 1,
                final_pos_acc,
                final_dx,
                final_dy,
                final_rot_acc,
                final_angle_err,
                *counts
            ])
            log(f"[AUTO] Loop {count+1} result → "
                f"PosAcc={final_pos_acc:.2f}%, ΔX={final_dx}, ΔY={final_dy}, "
                f"RotAcc={final_rot_acc:.2f}%, RotErr={final_angle_err:.2f}°", self)

            await asyncio.sleep(3.0)

            # 4) FINAL DEFAULT
            if confirmed_center:
                log(f"[AUTO] Loop {count+1}: DEFAULT_2 started ", self)
                home_mode = True
                stable_count_default = 0
                retry = 0
                while retry < 200:
                    dx, dy, dx_s, dy_s, cmd = calculate_steps(confirmed_center, HOME_DXDY)
                    if abs(dx) > TOLERANCE_PX or abs(dy) > TOLERANCE_PX:
                        await self.send_serial_command(cmd)
                        await asyncio.sleep(0.05)
                        retry += 1
                    else:
                        stable_count_default += 1
                        if stable_count_default >= STABLE_REQUIRED:
                            break
                        await asyncio.sleep(0.05)
                log(f"[AUTO] Loop {count+1}: DEFAULT_2 completed ", self)


            count += 1

        auto_mode = False
        send_enabled = False
        force_send = False
        log("[AUTO] AUTO_PAUSED", self)

    def toggle_target_mode(self):
        global setting_target_mode, align_done, rotation_done, Target_Point
        align_done = rotation_done = False
        self.reset_progress()

        if not setting_target_mode:
            setting_target_mode = True
            log("[TARGET] Click to TARGET", self)
            self.label_state.setText(self.t("Status") + ": " + self.t("MANUAL"))
            #self.label_message.setText(self.t("Click to set Target"))
        else:
            setting_target_mode = False
            Target_Point = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]
            log(f"[TARGET] TARGET_POINT reset to center → {Target_Point}", self)
            self.label_state.setText(self.t("Status") + ": " + self.t("STANDBY"))
            #self.label_message.setText(self.t("Standby"))
            self.update_target_point_label()

    def toggle_target_mode(self):
        global setting_target_mode, position_done, rotation_done, Target_Point
        position_done = rotation_done = False
        self.reset_progress()

        if not setting_target_mode:
            setting_target_mode = True
            log("[TARGET] Click to set TARGET_POINT", self)
        else:
            setting_target_mode = False
            Target_Point = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]
            log(f"[TARGET] TARGET_POINT reset to center → {Target_Point}", self)
            self.update_target_point_label()

    def default(self):
        global confirmed_center, default_mode, send_enabled, position_done, rotation_done

        position_done = rotation_done = False
        self.reset_progress()

        if not confirmed_center \
        or awaiting_done or sending_command \
        or send_enabled or auto_mode \
        or self.rotation_active:
            return

        default_mode = True
        log("[DEFAULT] DEFAULT started", self)

        dx, dy, dx_steps, dy_steps, command = calculate_steps(confirmed_center, HOME_DXDY)
        if dx_steps + dy_steps < 10:
            log("[DEFAULT] DEFAULT skipped", self)
            send_enabled = False
        else:
            asyncio.create_task(self.send_serial_command(command))
            log(f"[DEFAULT] Send command: {command}", self)

        asyncio.create_task(self.default_loop())

    def mousePressEvent(self, event):
        global Target_Point, setting_target_mode, last_sent_command, force_send, confirmed_center, send_enabled

        if setting_target_mode and event.button() == Qt.LeftButton:
            pos = event.pos()
            widget_pos = self.image_label.mapFrom(self, pos)
            x = widget_pos.x()
            y = widget_pos.y()

            if 0 <= x < self.image_label.width() and 0 <= y < self.image_label.height():
                Target_Point = [int(x), int(y)]
                self.update_target_point_label()
                log(f"[TARGET] TARGET_POINT set → {Target_Point}", self)

                last_sent_command = ""
                force_send = True
                setting_target_mode = False
                send_enabled = False

    def reset(self):
        global confirmed_center, stable_count, last_sent_command
        confirmed_center = None
        stable_count = 0
        last_sent_command = ""

    def update_frame(self):
        global confirmed_center, stable_count, angle, send_enabled, Target_Point, \
               default_mode, awaiting_done, sending_command, \
               has_f1, has_f2, Target_degree, position_done, rotation_done, force_send
        dx = dy = 0
        command = None
        if send_enabled and not confirmed_center:
            log("[DEBUG] 현재 confirmed_center 없음, 명령 대기중", self)

        if not self.serial_ready:
            print("[DEBUG] Serial not ready")
            return

        if sending_command:
            print("[DEBUG] Arduino command in progress")
        if awaiting_done:
            print("[DEBUG] Awaiting Arduino 'DONE' response")
        
        if position_done:
            self.align_bar.setValue(100)

        angle_error_val = self.initial_angle
        command = None

        if not self.serial_ready:
            return

        ret, frame = self.cap.read()
        if not ret:
            if not self.warned_once:
                log("[CAMERA] Received failed to frame", self)
                self.warned_once = True
        if self.mode == "IDLE":
            display = frame.copy()
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            self.image_label.setPixmap(pixmap)
            return

        if self.recording and self.video_writer:
            pixmap = self.grab()
            qimg = pixmap.toImage()
            tmp_img = qimg.convertToFormat(QImage.Format.Format_RGB888)
            width, height = tmp_img.width(), tmp_img.height()
            ptr = tmp_img.bits()
            ptr.setsize(height * width * 3)
            arr = np.array(ptr).reshape(height, width, 3)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            self.video_writer.write(arr)
            
        display = frame.copy()
        results = model(display, conf=0.4)
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY) # 화면 흑백
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # 

        if not (position_done or rotation_done):
            if not send_enabled and not self.rotation_active:
                self.label_coords.setText(self.t("Position_Err") + ": -")
                self.reset_progress()

        has_f1 = False
        has_f2 = False
        for box in results[0].boxes.data:
            _, _, _, _, _, cls = box.tolist()
            label = results[0].names[int(cls)]
            if label == "F1":
                has_f1 = True
            elif label == "F2":
                has_f2 = True

        wafer_type = None  # wafer
        if has_f1 and has_f2:
            wafer_type = "P100"
        elif has_f1:
            wafer_type = "P111"
        self.label_wafer_type.setText(self.t("Wafer Type") + ": " + str(wafer_type or "-")) # 웨이퍼 F1만 있으면 P111, F1, F2 있으면 P100 고정

        draw_yolo_boxes(display, results, self, wafer_type)

        if wafer_type in ("P100", "P111") and not self.summary_type_inited[wafer_type]: # self.summary_type_inited 초기화
            detection_stats[wafer_type] = {cls: [] for cls in detection_stats[wafer_type]}
            self.summary_type_inited[wafer_type] = True

        if wafer_type in ["P100", "P111"]: # summary png 저장 변수
            for box in results[0].boxes.data:
                _, _, _, _, conf, cls = box.tolist()
                label = results[0].names[int(cls)]
                if label in detection_stats[wafer_type]:
                    detection_stats[wafer_type][label].append(conf)

        wafer_center = None # wafer_center calculate
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

        if f1_center and wafer_center:
            current_angle_val = calculate_angle(f1_center, wafer_center)
            angle = current_angle_val
            diff = (Target_degree - current_angle_val + 540) % 360 - 180
            angle_error_val = abs(diff)
            self.label_angle_error.setText(self.t("Angle_Err") + f": {angle_error_val:.2f}°")
            cv2.putText(display, f"{current_angle_val:.0f}deg.", (f1_center[0] + 10, f1_center[1] - 10),
                        FONT, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)
        else:
            angle_error_val = self.initial_angle

        if wafer_center:
            confirmed_center = wafer_center
        else:
            confirmed_center = None
      
        if confirmed_center:
            dx, dy, dx_steps, dy_steps, command = calculate_steps(confirmed_center, Target_Point)
            cx, cy = confirmed_center
            self.label_coords.setText(self.t("Position_Err") + f":\n({dx}, {dy})")

        if send_enabled:
            self.update_progress(dx, dy, angle_error_val, mode="align")
        elif self.rotation_active:
            self.update_progress(dx, dy, angle_error_val, mode="rotation")

        if confirmed_center:
            cx, cy = confirmed_center
            cv2.putText(
                display,
                f"({cx},{cy})",
                (cx + 10, cy + 10),
                FONT, FONT_SCALE, (0, 255, 0), FONT_THICKNESS
            )
            if abs(dx) <= TOLERANCE_PX and abs(dy) <= TOLERANCE_PX:
                if send_enabled and not awaiting_done and not sending_command:
                    stable_count += 1
                    if stable_count >= STABLE_REQUIRED:
                        position_done = True
                        send_enabled = False
                        stable_count = 0
                        log("[POSITION]Position completed", self)
                        asyncio.create_task(self.send_serial_command("STOP_ALIGN"))
                        return
            else:
                stable_count = 0

            if send_enabled \
                and stable_count < STABLE_REQUIRED \
                and not awaiting_done and not sending_command \
                and command is not None \
                and (command != last_sent_command or force_send):
                    asyncio.create_task(self.send_serial_command(command))
                    force_send = False
        else:
            self.label_coords.setText(self.t("Position_Err") + ": -")
            if not (position_done or rotation_done):
                self.reset_progress()

            if send_enabled:
                self.update_progress(dx, dy, angle_error_val, mode="position")
            elif self.rotation_active:
                self.update_progress(dx, dy, angle_error_val, mode="rotation")

            if abs(dx) <= TOLERANCE_PX and abs(dy) <= TOLERANCE_PX:
                if send_enabled and not awaiting_done and not sending_command:
                    stable_count += 1
                    if stable_count >= STABLE_REQUIRED:
                        position_done = True
                        send_enabled = False
                        stable_count = 0
                        asyncio.create_task(self.send_serial_command("STOP_ALIGN"))
                        return
            else:
                stable_count = 0

            if send_enabled \
                and stable_count < STABLE_REQUIRED \
                and not awaiting_done and not sending_command \
                and command is not None \
                and (command != last_sent_command or force_send):
                    asyncio.create_task(self.send_serial_command(command))
                    force_send = False

        if not self.rotation_active and not default_mode:
            cv2.circle(display, tuple(Target_Point), 5, (0, 0, 255), -1)
            if confirmed_center:
                cv2.circle(display, confirmed_center, 5, (0, 255, 0), -1)
                cv2.line(display, tuple(Target_Point), confirmed_center, (255, 0, 0), 2)

        self.label_wafer_type.setText(self.t("Wafer Type") + ": " + str(wafer_type or "-"))

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
            h, w, ch = zoom_crop.shape
            bytes_per_line = ch * w
            qimg = QImage(zoom_crop.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
            zoomed_qimg = qimg.scaled(400, 300, Qt.KeepAspectRatio)

            if hasattr(self, 'zoom_window') and self.zoom_window:
                self.zoom_window.zoom_label.setPixmap(QPixmap.fromImage(zoomed_qimg))

        if (
            send_enabled 
            and stable_count < STABLE_REQUIRED 
            and not awaiting_done and not sending_command 
            and not sending_command 
            and command != last_sent_command  
            and not force_send             
        ):
            asyncio.create_task(self.send_serial_command(command))

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F1:
            self.capture_action # F1 누르면 캡쳐
        elif key == Qt.Key_F2:
            self.record_action # F2 누르면 녹화

    def capture_screenshot(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        pixmap = self.grab()
        pixmap.save(filename)
        QMessageBox.information(self, self.t("Screenshot"), self.t(f"Saved as {filename}"))

    def show_help_dialog(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("도움말")
        msg.setText("추후 기재") # 수정
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
        self.setWindowTitle("Log")
        self.setGeometry(200, 200, 600, 400)
        self.logs = []
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.filter_box = QComboBox()
        self.filter_box.addItems(["ALL", "ALIGN", "ROTATION", "CHANGED", "ERROR", "Status"]) #추후에 추가
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

    def save_log_txt(self):
        try:
            target_dir = r"C:\Users\Mai\Desktop\wafer_project\log and summary"
            os.makedirs(target_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            log_filename = os.path.join(target_dir, f"log_{timestamp}.txt")
            with open(log_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(self.logs))

            for wafer_type, stats in detection_stats.items():
                LogWindow.save_detection_summary_chart(
                    wafer_type, stats, timestamp, target_dir
                )

            QMessageBox.information(
                self, "Saved",
                f"Logs → {log_filename}\n"
                f"Charts saved in {target_dir}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save logs:\n{e}")

    def save_detection_summary_chart(wafer_type, stats, timestamp, target_dir):
        classes = [c for c, lst in stats.items() if lst]
        if not classes:
            return  

        counts      = [len(stats[c]) for c in classes]
        confidences = [round(sum(stats[c]) / len(stats[c]), 2) for c in classes]

        fig, ax1 = plt.subplots(figsize=(8,5))
        ax1.set_title(f"{wafer_type} Detection Summary", pad=20, fontsize=14)
        ax1.bar(classes, counts)
        ax1.set_ylabel("Count")

        ax2 = ax1.twinx()
        ax2.plot(classes, confidences, marker="o", color="red")
        ax2.set_ylabel("Confidence")
        # 오탐지 그래프
        mis = [c for c in classes if c != wafer_type]
        correct = [c for c in classes if c == wafer_type]
        ax1.bar(correct,   [len(stats[c]) for c in correct])
        ax1.bar(mis,       [len(stats[c]) for c in mis], alpha=0.5, hatch='//')

        for i, conf in enumerate(confidences):
            ax2.text(i, conf + 0.02, f"{conf:.2f}", 
                    ha="center", va="bottom", fontsize=9)

        fig.tight_layout(rect=[0,0,1,0.92])

        png_name      = f"{wafer_type}_{timestamp}_detection_summary.png"
        full_png_path = os.path.join(target_dir, png_name)
        plt.savefig(full_png_path)
        plt.close()

        fig.tight_layout() # 비율 딱 맞게

class VariableTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.default_values = {
            "HOME_DXDY": "(556, 188)", # HOME_DXDY 먼저 수정
            "Target_Point": "[640, 360]",
            "PIXEL_TO_STEP": "{'x': 33.0, 'y': 45.65}", 
            "TOLERANCE_PX": "1",
            "TOLERANCE_R": "1",
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

        apply_button = QPushButton(self.main_window.t("Apply"))
        apply_button.clicked.connect(self.apply_changes)
        reset_button = QPushButton(self.main_window.t("Reset"))
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
                    if name == "Target_Point":
                        self.main_window.update_target_point_label()
            update_variable("HOME_DXDY", eval)
            update_variable("Target_Point", eval)
            update_variable("PIXEL_TO_STEP", eval)
            update_variable("TOLERANCE_PX", int)
            update_variable("TOLERANCE_R", int,)
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
        self.setWindowTitle(self.t("Point_Zoom"))
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
        qimg = qimg.rgbSwapped()
        self.zoom_label.setPixmap(QPixmap.fromImage(qimg))

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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"log_{timestamp}.txt"
            with open(log_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(self.logs))

            for wafer_type, stats in detection_stats.items():
                LogWindow.save_detection_summary_chart(wafer_type, stats, timestamp)

            QMessageBox.information(
                self,
                "Saved",
                f"Log saved as {log_filename}\n"
                f"Summary charts saved as *_{timestamp}_detection_summary.png"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save logs:\n{e}")

def main():
    app = QApplication(sys.argv)
    splash_pix = QPixmap("C:/Users/Mai/Desktop/wafer_project/splash.png").scaled(
        1400, 576, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )
    splash = QSplashScreen(splash_pix)
    splash.show()
    app.processEvents()
    loop = QEventLoop(app)              
    asyncio.set_event_loop(loop)      
    def start_main_program():
        splash.close()
        window = MainWindow(loop)      
        window.loop = loop              
        window.serial_ready = True     
    QTimer.singleShot(3000, start_main_program)
    with loop:
        loop.run_forever()            
if __name__ == "__main__":
    main()
