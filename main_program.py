import sys, os, gc, time, asyncio, cv2, torch, numpy as np, serial_asyncio 
from ultralytics import YOLO
from qasync import QEventLoop
from PyQt5.QtCore import Qt, QTimer 
from PyQt5.QtGui import QImage, QPixmap 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit, QInputDialog, QGroupBox,
    QVBoxLayout, QHBoxLayout, QFormLayout, QScrollArea, QComboBox, QSizePolicy, QLineEdit, QMessageBox, QTabWidget,
    QAction, QMenu, QSplashScreen, QGridLayout, QProgressBar, QCheckBox, QFileDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
import matplotlib.pyplot as plt
import random
import csv

model = YOLO("C:\\Users\\Mai\\Desktop\\wafer_project\\redline.pt") # best.pt check
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
Target_Point = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]
HOME_DXDY = (600, 200)
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
AUTO_LOOP_COUNT = 0  # 0이면 무한, 1부터 ~
RANDOM_X_MIN, RANDOM_X_MAX = 400, 800
RANDOM_Y_MIN, RANDOM_Y_MAX = 300, 500
RANDOM_DEG_MIN, RANDOM_DEG_MAX = 0, 360

CLASS_COLORS = {
    'F1': (0, 0, 50), 'F2': (0, 0, 100),
    'P100': (0, 0, 255), 'P111': (255, 0, 0),
}
detection_stats = {
    "P100": {"F1": [], "F2": [], "P100": [], "P111": []},
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
align_done = False
rotation_done = False
position_accuracy_list = []
rotation_accuracy_list = []
RANDOM_MODE = False

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

def calculate_accuracy_px(target_point, center_point, initial_offset=(FRAME_WIDTH, FRAME_HEIGHT)):
    dx = target_point[0] - center_point[0]
    dy = target_point[1] - center_point[1]
    error_distance = (dx ** 2 + dy ** 2) ** 0.5
    start_distance = (initial_offset[0] ** 2 + initial_offset[1] ** 2) ** 0.5

    if start_distance == 0:
        return 100.0
    else:
        acc = (1 - error_distance / start_distance) * 100
        if acc < 0: acc = 0
        if acc > 100: acc = 100
        return round(acc, 2)

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
    # dx = pt2[0] - pt1[0]
    # dy = pt2[1] - pt1[1]
    # angle = np.degrees(np.arctan2(dy, dx))
    # return angle + 360 if angle < 0 else angle
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    raw = np.degrees(np.arctan2(dy, dx))
    if raw < 0:
        raw += 360
    return (180 - raw) % 360

def calculate_rotation(current_angle, target_angle):
    # dR = (target_angle - current_angle + 540) % 360 - 180
    # diR = 1
    # stR = max(1, min(int(abs(dR) * DEGREE_TO_STEP), MAX_ROTATION_STEP))
    # return diR, stR
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
        self.setWindowTitle("Wafer_Aligner")
        self.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT + 200 + self.menuBar().height())
        self.setMouseTracking(True)
        menubar = self.menuBar()

        self.recording = False
        self.video_writer = None
        window = None
        self.position_completed_shown = False
        self.rotation_completed_shown = False
        self.message_timer = QTimer(self)
        self.message_timer.setSingleShot(True)
        self.message_timer.timeout.connect(self.set_standby_message)

        self.label_wafer_type      = QLabel()
        self.label_target_point    = QLabel()
        self.label_coords          = QLabel()
        self.label_target_degree   = QLabel()
        self.label_angle_error     = QLabel()

        self.language = "en"
        self.texts = {
            "ko": {
                "RANDOM MODE": "무작위",
                "align": "자동모드",
                "Align": "자동모드",
                "ALGIN": "자동모드",
                "set target": "목표설정",
                "Set_Target": "목표설정",
                "SET_TARGET": "목표설정",
                "SET TARGET": "목표설정",
                "set_target": "목표설정",
                "HOMING": "원위치",
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
                "POSITION ACCURACY(%)": "정렬 정확도(%)",
                "ROTATION ACCURACY(%)": "회전 정확도(%)",
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
                "HOMING": "원위치중",
                "AUTO": "자동모드",
                "MANUAL": "수동모드",
                "Operation Not Allowed": "작업 불가",
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
                "RANDOM MODE": "RANDOM MODE",
                "log": "Log",
                "Status": "Status",
                "STANDBY": "STANDBY",
                "ALIGNING": "ALIGNING",
                "ROTATING": "ROTATING",
                "HOMING": "HOMING",
                "AUTO": "AUTO",
                "MANUAL": "MANUAL",
                "Standby": "Standby",
                "Positioning...": "Positioning...",
                "Rotationing...": "Rotationing...",
                "Homing...": "Homing...",
                "Align...": "Align...",
                "Click to set Target": "Click to set Target",
                "Function": "Function",
                "Settings": "Settings",
                "ROTATION ACCURACY(%)": "ROTATION ACCURACY(%)",
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

        self.function_menu = menubar.addMenu(self.t("Function"))
        self.align_action = QAction(self.t("Align"), self, triggered=self.show_advanced_tab)
        self.position_action = QAction(self.t("Position"), self, triggered=self.show_align_tab)
        self.rotation_action = QAction(self.t("Rotation"), self, triggered=self.show_angle_tab)
        self.function_menu.addAction(self.align_action)
        self.function_menu.addAction(self.position_action)
        self.function_menu.addAction(self.rotation_action)

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
        #self.label_message = QLabel(self.t(" Standby "))
        #self.label_message.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.align_label = QLabel(self.t("POSITION ACCURACY(%)"))
        self.align_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.rotation_label = QLabel(self.t("ROTATION ACCURACY(%)"))
        self.rotation_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.align_bar = QProgressBar()
        self.rotation_bar = QProgressBar()
        self.align_bar.setMaximum(100)
        self.rotation_bar.setMaximum(100)
        progress_layout = QVBoxLayout()
        #progress_layout.addWidget(self.label_message)
        progress_layout.addWidget(self.align_label)
        progress_layout.addWidget(self.align_bar)
        progress_layout.addWidget(self.rotation_label)
        progress_layout.addWidget(self.rotation_bar)
        self.progress_widget = QWidget()
        self.progress_widget.setLayout(progress_layout)
        self.progress_widget.setFixedWidth(300)
        self.label_wafer_type.setText(self.t("Wafer Type") + ":\n-")
        self.label_target_point.setText(self.t("Target Point:") + f"\n({Target_Point[0]}, {Target_Point[1]})")
        self.label_coords.setText(self.t("Position_Err") + ":\n" + "(0, 0)")
        self.label_state = QLabel("Status: STANDBY") # Status 추후 삭제
        self.label_state.setStyleSheet("""
            font-size: 16px; 
            color: transparent;
            background: transparent;
            border: none;
        """)
        self.label_target_degree.setText(self.t("Target Deg.") + f": {Target_degree}°")
        self.label_angle_error.setText(self.t("Angle_Err") + f": 0°")
        for lbl in [self.label_wafer_type, self.label_target_point, self.label_coords,
                    self.label_target_degree, self.label_angle_error]:
            lbl.setStyleSheet("font-size: 15px; padding: 10px;")

        status_layout = QGridLayout()
        status_layout.addWidget(self.label_wafer_type, 0, 0)
        status_layout.addWidget(self.label_target_point, 0, 1)
        status_layout.addWidget(self.label_coords, 0, 2)
        status_layout.addWidget(self.label_target_degree, 1, 1)
        status_layout.addWidget(self.label_angle_error, 1, 2)
        status_layout.addWidget(self.label_state, 1, 0)
        self.label_state = QLabel("Status: STANDBY")
        self.label_state.setStyleSheet("""
            font-size: 16px; 
            color: transparent;
            background: transparent;
            border: none;
        """)

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
        asyncio.create_task(self.serial_setup())
        self.state = "Standby"

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

    def toggle_recording(self, checked):
        if checked:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"record_{timestamp}.avi"
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT)
            )
            self.recording = True
            #self.label_message.setText(self.t("Recording started"))
            log(f"[RECORD] Recording started: {filename}", self)
        else:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            #self.label_message.setText("Recording completed")
            log(self.t(f"[RECORD] Recording completed"), self)

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
            self.exit_action.setText(self.t("Exit"))
            self.function_menu.setTitle(self.t("Function"))
            self.align_action.setText(self.t("Align"))
            self.position_action.setText(self.t("Position"))
            self.rotation_action.setText(self.t("Rotation"))
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
            self.tabs.setTabText(self.tabs.indexOf(self.advanced_tab), self.t("Align"))

            # 버튼/라벨
            self.m_button.setText(self.t("SET TARGET"))
            self.s_button.setText(self.t("START"))
            self.u_button.setText(self.t("DEFAULT"))
            self.a_button.setText(self.t("SET TARGET"))
            self.r_button.setText(self.t("START"))
            self.t_button.setText(self.t("Align"))
            self.align_label.setText(self.t("POSITION ACCURACY(%)"))
            self.rotation_label.setText(self.t("ROTATION ACCURACY(%)"))
            #self.label_message.setText(self.t("Standby"))
            self.label_wafer_type.setText(self.t("Wafer Type") + ": -")
            self.label_coords.setText(self.t("Position_Err") + ":\n" + "(0, 0)")
            self.label_target_point.setText(self.t("Target Point") + f":\n({Target_Point[0]}, {Target_Point[1]})")
            self.label_target_degree.setText(self.t("Target Deg.") + f":{Target_degree}°")
            self.label_angle_error.setText(self.t("Angle_Err") + f":0°")
            self.label_state.setText(self.t("Status") + ": " + self.t("STANDBY"))
        except Exception as e:
            print("update_all_texts 오류:", e)

    def update_target_point_label(self):
        self.label_target_point.setText(self.t("Target Point") + f":\n({Target_Point[0]}, {Target_Point[1]})")

    def apply_auto_var_changes(self):
        try:
            global HOME_DXDY, Target_Point, Target_degree, AUTO_LOOP_COUNT, RANDOM_MODE, RANDOM_DEG_MAX, RANDOM_DEG_MIN, RANDOM_X_MAX, RANDOM_X_MIN, RANDOM_Y_MAX, RANDOM_Y_MIN
            HOME_DXDY     = eval(self.auto_var_fields["HOME_DXDY"].text())
            Target_Point  = eval(self.auto_var_fields["TARGET_POINT"].text())
            Target_degree = float(self.auto_var_fields["TARGET_DEGREE"].text())
            AUTO_LOOP_COUNT = int(self.auto_var_fields["LOOP"].text())
            RANDOM_X_MIN      = int(self.auto_var_fields["RANDOM_X_MIN"].text())
            RANDOM_X_MAX      = int(self.auto_var_fields["RANDOM_X_MAX"].text())
            RANDOM_Y_MIN      = int(self.auto_var_fields["RANDOM_Y_MIN"].text())
            RANDOM_Y_MAX      = int(self.auto_var_fields["RANDOM_Y_MAX"].text())
            RANDOM_DEG_MIN    = float(self.auto_var_fields["RANDOM_DEG_MIN"].text())
            RANDOM_DEG_MAX    = float(self.auto_var_fields["RANDOM_DEG_MAX"].text())

            RANDOM_MODE = self.random_checkbox.isChecked()

            self.update_target_point_label()
            self.label_target_degree.setText(self.t("Target Deg.") + f":{Target_degree}°")
            #log("[CHANGED] AUTO 변수 적용됨", self)
        except Exception as e:
            #log(f"[ERROR] AUTO 변수 적용 실패: {e}", self)
            show_warning(self, "Apply Error", str(e))

    def reset_auto_var_fields(self):
        self.auto_var_fields["HOME_DXDY"].setText(str((556, 188)))
        self.auto_var_fields["TARGET_POINT"].setText(str([FRAME_WIDTH // 2, FRAME_HEIGHT // 2]))
        self.auto_var_fields["TARGET_DEGREE"].setText(str(0))
        # 방금 수정한 함수 호출
        self.apply_auto_var_changes()

    def set_standby_message(self):
        #self.label_message.setText(self.t("Standby"))
        self.position_completed_shown = False
        self.rotation_completed_shown = False

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
        self.u_button.clicked.connect(self.move_home)

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

        # Advanced 탭
        self.advanced_tab = QWidget()
        adv_layout = QVBoxLayout()

        self.t_button = QPushButton(self.t("Align"))    
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
            "LOOP": str(AUTO_LOOP_COUNT),
            "RANDOM_X_MIN": str(RANDOM_X_MIN),
            "RANDOM_X_MAX": str(RANDOM_X_MAX),
            "RANDOM_Y_MIN": str(RANDOM_Y_MIN),
            "RANDOM_Y_MAX": str(RANDOM_Y_MAX),
            "RANDOM_DEG_MIN": str(RANDOM_DEG_MIN),
            "RANDOM_DEG_MAX": str(RANDOM_DEG_MAX),
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

        self.advanced_tab.setLayout(adv_layout)

        self.variable_tab = VariableTab(self)

        # Variable 탭
        self.variable_tab = VariableTab(self)


    def start_auto_align(self):
        global auto_mode, rotation_done, align_done
        if send_enabled or self.rotation_active or home_mode:
            #log("[ERROR] Cannot start AUTO now", self)
            self.show_warning(self.t("Operation Not Allowed", "AUTO cannot be excuted at this time"))
            return
        #self.update_state(self.t("AUTO"))
        rotation_done = align_done = False
        self.reset_progress()
        auto_mode = True
        if self.loop and not self.loop.is_closed():
            #log("[AUTO] Started", self)
            self.loop.create_task(self.auto_align_loop())
        else:
            log("[ERROR] Event loop is not ready or already closed", self)

    def show_align_tab(self):
        if self.tabs.indexOf(self.align_tab) == -1:
            self.tabs.addTab(self.align_tab, self.t("Position"))
        self.tabs.show()
        self.tabs.setCurrentWidget(self.align_tab)

    def show_angle_tab(self):
        global align_done, rotation_done
        align_done = rotation_done = False
        if self.tabs.indexOf(self.angle_tab) == -1:
            self.tabs.addTab(self.angle_tab, self.t("Rotation"))
        self.tabs.show()
        self.tabs.setCurrentWidget(self.angle_tab)

    def show_advanced_tab(self):
        if self.tabs.indexOf(self.advanced_tab) == -1:
            self.tabs.addTab(self.advanced_tab, self.t("Align"))
        self.tabs.show()
        self.tabs.setCurrentWidget(self.advanced_tab)

    def update_progress(self, dx, dy, angle_error, mode):
        if mode == "align":

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

    def apply_align_var_changes(self):
        try:
            global HOME_DXDY, Target_Point, Target_degree
            HOME_DXDY = eval(self.align_var_fields["HOME_DXDY"].text())
            Target_Point = eval(self.align_var_fields["TARGET_POINT"].text())
            Target_degree = float(self.align_var_fields["TARGET_DEGREE"].text())
            self.update_target_point_label()
            self.label_target_degree.setText(self.t("Target Deg.") + f":{Target_degree}°")
            log("[CHANGED] Align 변수 적용됨", self)
        except Exception as e:
            log(f"[ERROR] Align 변수 적용 실패: {e}", self)
            self.show_warning("Apply Error", str(e))

    def reset_align_var_fields(self):
        self.align_var_fields["HOME_DXDY"].setText(str((556, 188)))
        self.align_var_fields["TARGET_POINT"].setText(str([FRAME_WIDTH // 2, FRAME_HEIGHT // 2]))
        self.align_var_fields["TARGET_DEGREE"].setText(str(0))
        self.apply_align_var_changes()

    def reset_progress(self):
        self.align_bar.setValue(0)
        self.rotation_bar.setValue(0)

    def set_target_degree(self):
        if send_enabled or auto_mode or home_mode or self.rotation_active:
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
        #self.update_state("Standby")

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

    def set_waiting(self):
        self.label_state.setText(self.t("Status") + ": " + self.t("STANDBY"))
        if not align_done:
            self.label_state.setText(self.t("Status") + ": " + self.t("STANDBY"))

    async def serial_setup(self):
        try:
            self.ser_reader, self.ser_writer = await serial_asyncio.open_serial_connection(
                url=SERIAL_PORT, baudrate=BAUDRATE)
            self.serial_ready = True
            log("[INIT] Serial ready", self)
        except Exception as e:
            log(f"[ERR] Serial setup failed: {e}", self)

    async def move_home_loop(self):
        global confirmed_center, home_mode, dx, dy, send_enabled
        self.reset_progress()
        if confirmed_center:
            home_mode = True
            log("[HOMING] Start HOMING", self)
            #self.label_message.setText("HOMING...")
            retry = 0
            while retry < 10:
                dx, dy, dx_steps, dy_steps, command = calculate_steps(confirmed_center, HOME_DXDY)
                if dx_steps + dy_steps < 10:
                    log("[HOMING] HOMING completed with acceptable tolerance", self)
                    break
                await self.send_serial_command(command)
                await asyncio.sleep(0.3)
                retry += 1
            home_mode = False
            #self.label_state.setText("Status: HOMING_DONE")
            log("[HOMING] Final confirmed_center → {}".format(confirmed_center), self)


    async def send_serial_command(self, command):
        global sending_command, awaiting_done, last_sent_command, force_send, home_mode, send_enabled

        if not self.serial_ready or self.ser_writer is None:
            return

        critical_stop = command.upper() in ("STOP", "STOP_ALIGNMENT")
        if (sending_command or awaiting_done) and not critical_stop:
            return
        if command == last_sent_command and not force_send and not critical_stop:
            return

        sending_command = True
        awaiting_done = True

        if home_mode:
            self.label_state.setText(self.t("Status") + ": " + self.t("HOMING"))
        elif command == "STOP_ALIGNMENT":
            self.label_state.setText(self.t("Status") + ": " + self.t("ALIGN_STOPPING"))
        else:
            actual = command
            if command.upper() == "STOP" or command == "STOP_ALIGNMENT":
                actual = "0,0,0,0,0,0"

            parts = actual.split(",")
            if len(parts) == 6 and (int(parts[4]) != 0 or int(parts[5]) != 0):
                self.label_state.setText("Status: ROTATING")
                log(f"[ROTATION] Sending ROTATION command: {actual}", self)
            else:
                self.label_state.setText("Status: ALIGNING")
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
                    if home_mode:
                        self.label_state.setText("Status: HOMING_TIMEOUT")
                    elif command == "STOP_ALIGNMENT":
                        self.label_state.setText("Status: ALIGN_STOP_TIMEOUT")
                    else:
                        self.label_state.setText("Status: CMD_TIMEOUT")
                    break

        except Exception as e:
            log(f"[ERR] Serial communication error: {e}", self)
            self.label_state.setText(self.t("Status") + ": " + self.t("SERIAL_ERROR"))

        finally:
            sending_command = False
            awaiting_done = False

            if home_mode:
                self.label_state.setText(self.t("Status") + ": " + self.t("DEFAULT_DONE"))
                home_mode = False
                send_enabled = False

            elif last_sent_command == "0,0,0,0,0,0":
                self.label_state.setText(self.t("Status") + ": " + self.t("STANDBY"))
            else:
                parts = last_sent_command.split(",")
                if len(parts) == 6 and (int(parts[4]) != 0 or int(parts[5]) != 0):
                    self.label_state.setText(self.t("Status") + ": " + self.t("ROTATION_STEP_DONE"))
                else:
                    self.label_state.setText(self.t("Status") + ": " + self.t("ALIGN_STEP_DONE"))

    # def update_state(self, new_state):
    #     self.state = new_state
    #     self.label_state.setText(f"{self.t('Status')}: {self.t(new_state)}")
    #     if new_state == "STANDBY":
    #         self.label_message.setText(self.t("Standby"))
    #     elif new_state == "ALIGNING":
    #         self.label_message.setText(self.t("Positioning..."))
    #     elif new_state == "ROTATING":
    #         self.label_message.setText(self.t("Rotating..."))
    #     elif new_state == "DEFAULTING":
    #         self.label_message.setText(self.t("DEFAULTING..."))
    #     elif new_state == "ALIGN":
    #         self.label_message.setText(self.t("Align..."))
    #     elif new_state == "Set Target":
    #         self.label_message.setText(self.t("Click to Target"))

    def toggle_log_window(self):
        if self.log_window.isVisible():
            self.log_window.hide()
        else:
            self.log_window.show()
            self.log_window.raise_()

    def toggle_send(self):  # ALIGN 함수
        global send_enabled, stable_count, Target_Point, auto_mode, align_done, rotation_done

        if self.rotation_active or auto_mode or home_mode:
            log("[ERROR] Cannot start ALIGN during other operation", self)
            self.show_warning("Operation Not Allowed", "ALIGN cannot be excuted at this time")
            return

        if send_enabled:
            send_enabled = False
            self.reset_progress()
            #self.update_state("STANDBY")
            asyncio.create_task(self.send_serial_command("STOP_ALIGNMENT"))
            log("[ALIGN] ALIGN stopped", self)
            return

        align_done = rotation_done = False
        self.reset_progress()
        send_enabled = True
        auto_mode = False

        log("[ALIGN] ALIGN started", self)
        stable_count = 0
        self.warned_once = False

        # ★ 정렬 시작 직전 초기 오프셋(초기 거리) 반드시 갱신
        if confirmed_center:
            dx0 = abs(Target_Point[0] - confirmed_center[0])
            dy0 = abs(Target_Point[1] - confirmed_center[1])
            self.initial_offset = (dx0, dy0)
        else:
            self.initial_offset = (FRAME_WIDTH, FRAME_HEIGHT)

        #self.update_state("ALIGNING")
    async def start_rotation(self):
        global angle, rotation_done
        if send_enabled or home_mode or self.rotation_active:
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
        #self.update_state("ROTATING")
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

            # ③ 프로그레스 계산 (0→360 구간 그대로 반영)
            traveled = (angle - self.rotation_start + 360) % 360
            prog     = min(traveled / self.rotation_total, 1.0) * 100
            self.rotation_bar.setValue(int(prog))

            # ④ UI 업데이트 # 수정
            #self.label_angle.setText(f"Angle: {angle:.0f}°")
            self.label_angle_error.setText(f"Angle_Err: {angle_error:.2f}°")

            # ⑤ 완료 판정
            if angle_error < TOLERANCE_R:
                rotation_done = True
                log("[ROTATION] Rotation completed", self)
                await self.send_serial_command("STOP")
                self.rotation_active = False
                #self.update_state("STANDBY")
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
            #self.update_state("STANDBY")  

    def reload_yolo_model(self):
        global model
        log("[YOLO] reload model...", self)
        #self.label_message.setText("Initializing model...")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = YOLO("redline.pt")
        log("[YOLO]Successfully reload model", self)
        #self.label_message.setText("Model initialization completed")

    async def auto_align_loop(self):
        global auto_mode, send_enabled, home_mode, force_send, confirmed_center
        global align_done, rotation_done, AUTO_LOOP_COUNT, Target_Point, Target_degree

        log("[AUTO] AUTO started", self)
        auto_mode = True
        count = 0

        while auto_mode and (AUTO_LOOP_COUNT == 0 or count < AUTO_LOOP_COUNT):
            # ── 랜덤 모드 타겟 설정 ──
            if RANDOM_MODE:
                Target_Point = [
                    random.randint(RANDOM_X_MIN, RANDOM_X_MAX),
                    random.randint(RANDOM_Y_MIN, RANDOM_Y_MAX)
                ]
                Target_degree = random.uniform(RANDOM_DEG_MIN, RANDOM_DEG_MAX)
                self.update_target_point_label()
                self.label_target_degree.setText(
                    self.t("Target Deg.") + f": {Target_degree:.1f}°"
                )
                log(f"[AUTO] New random target → Point: {Target_Point}, Degree: {Target_degree:.1f}°", self)
                await asyncio.sleep(0.2)

            # 1) DEFAULT 단계
            if confirmed_center:
                home_mode = True
                log(f"[AUTO] Loop {count+1}: DEFAULT", self)
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

            # 2) ALIGN 단계
            log(f"[AUTO] Loop {count+1}: ALIGN", self)
            align_done = False
            stable_count = 0
            if confirmed_center:
                dx0 = abs(Target_Point[0] - confirmed_center[0])
                dy0 = abs(Target_Point[1] - confirmed_center[1])
                self.initial_offset = (dx0, dy0)
            else:
                self.initial_offset = (FRAME_WIDTH, FRAME_HEIGHT)

            position_accuracy_list.clear()
            retry = 0
            while retry < 200:
                await asyncio.sleep(0.05)
                if confirmed_center:
                    dx, dy, dx_s, dy_s, cmd = calculate_steps(confirmed_center, Target_Point)
                    pos_acc = calculate_accuracy_px(Target_Point, confirmed_center)
                    position_accuracy_list.append(pos_acc)
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
            final_pos_acc = position_accuracy_list[-1] if position_accuracy_list else 0.0
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
    def move_home(self):
        global confirmed_center, home_mode, send_enabled, align_done, rotation_done

        align_done = rotation_done = False
        self.reset_progress()

        if not confirmed_center \
        or awaiting_done or sending_command \
        or send_enabled or auto_mode \
        or self.rotation_active:
            return

        home_mode = True
        #self.update_state("HOMING")
        log("[HOMING] Start HOMING", self)

        dx, dy, dx_steps, dy_steps, command = calculate_steps(confirmed_center, HOME_DXDY)
        if dx_steps + dy_steps < 10:
            #self.update_state("STANDBY")
            log("[HOMING] HOMING skipped", self)
            send_enabled = False
        else:
            asyncio.create_task(self.send_serial_command(command))
            log(f"[HOMING] Send command: {command}", self)

        asyncio.create_task(self.move_home_loop())

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
                self.label_state.setText("Status: STANDBY")

    def reset(self):
        global confirmed_center, stable_count, last_sent_command
        confirmed_center = None
        stable_count = 0
        last_sent_command = ""

    def update_frame(self):
        global confirmed_center, stable_count, angle, send_enabled, Target_Point, \
            home_mode, awaiting_done, sending_command, has_f1, has_f2, \
            Target_degree, align_done, rotation_done, force_send, auto_mode

        # 1) 매 프레임마다 상태창 Target_Point 라벨 갱신
        self.update_target_point_label()

        # 2) 디버그 / 시리얼 체크
        if send_enabled and not confirmed_center:
            log("[DEBUG] 현재 confirmed_center 없음, 명령 대기중", self)
        if not self.serial_ready:
            print("[DEBUG] Serial not ready")
            return
        if sending_command:
            print("[DEBUG] Arduino command in progress")
        if awaiting_done:
            print("[DEBUG] Awaiting Arduino DONE response")
        # if align_done and not auto_mode:
        #     self.label_message.setText("POSITION completed")
 

        # 3) 카메라 프레임 읽기
        ret, frame = self.cap.read()
        if not ret:
            if not self.warned_once:
                log("[CAMERA] Received failed to frame", self)
                self.warned_once = True
            return

        # 4) 녹화 중이면 캡처 저장
        if self.recording and self.video_writer:
            pixmap = self.grab()
            qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits()
            ptr.setsize(h * w * 3)
            arr = np.array(ptr).reshape(h, w, 3)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            self.video_writer.write(arr)

        # 5) 그레이스케일 → BGR 처리
        display = frame.copy()
        results = model(display, conf=0.4)
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 6) 기본 상태 초기화
        if not (align_done or rotation_done or auto_mode or send_enabled or self.rotation_active or home_mode):
            self.label_coords.setText(self.t("Position_Err") + ": -")
            self.reset_progress()

        # 7) F1/F2 감지 및 박스 그리기
        has_f1 = has_f2 = False
        for box in results[0].boxes.data:
            _, _, _, _, _, cls = box.tolist()
            label = results[0].names[int(cls)]
            if label == "F1": has_f1 = True
            elif label == "F2": has_f2 = True

        wafer_type = None
        if has_f1 and has_f2: wafer_type = "P100"
        elif has_f1:          wafer_type = "P111"
        self.label_wafer_type.setText(self.t("Wafer Type") + ": " + str(wafer_type or "-"))
        draw_yolo_boxes(display, results, self, wafer_type)

        # 8) 통계 업데이트
        if wafer_type in ("P100", "P111") and not self.summary_type_inited[wafer_type]:
            detection_stats[wafer_type] = {c: [] for c in detection_stats[wafer_type]}
            self.summary_type_inited[wafer_type] = True
        if wafer_type in ("P100", "P111"):
            for box in results[0].boxes.data:
                _, _, _, _, conf, cls = box.tolist()
                lbl = results[0].names[int(cls)]
                if lbl in detection_stats[wafer_type]:
                    detection_stats[wafer_type][lbl].append(conf)

        # 9) 중심점 계산
        wafer_center = None
        f1_center = None
        for box in results[0].boxes.data:
            x1, y1, x2, y2, *_ = box.tolist()
            cls = int(box[5])
            lbl = results[0].names[cls]
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            if lbl in ("P100","P111"): wafer_center = (cx, cy)
            elif lbl == "F1":         f1_center = (cx, cy)

        # 10) 각도 오차 표시
        if f1_center and wafer_center:
            cur_ang = calculate_angle(f1_center, wafer_center)
            angle = cur_ang
            diff = (Target_degree - cur_ang + 540) % 360 - 180
            err = abs(diff)
            self.label_angle_error.setText(self.t("Angle_Err") + f": {err:.2f}°")
            cv2.putText(display, f"{cur_ang:.0f}deg.",
                        (f1_center[0]+10, f1_center[1]-10),
                        FONT, FONT_SCALE, (0,255,255), FONT_THICKNESS)
        else:
            err = self.initial_angle

        confirmed_center = wafer_center


        if send_enabled and confirmed_center:
            acc = calculate_accuracy_px(Target_Point, confirmed_center, initial_offset=self.initial_offset)
            self.align_bar.setValue(int(acc))
            dx, dy, dx_s, dy_s, cmd = calculate_steps(confirmed_center, Target_Point)
            if abs(dx)<=TOLERANCE_PX and abs(dy)<=TOLERANCE_PX:
                stable_count +=1
                if stable_count>=STABLE_REQUIRED:
                    align_done=True; send_enabled=False; stable_count=0
                    #self.label_message.setText(" POSITION completed ")
                    self.align_bar.setValue(100)
                    asyncio.create_task(self.send_serial_command("STOP_ALIGN"))
            else:
                stable_count=0
            if (send_enabled and stable_count<STABLE_REQUIRED
                and not awaiting_done and not sending_command
                and cmd and (cmd!=last_sent_command or force_send)):
                asyncio.create_task(self.send_serial_command(cmd))
                force_send=False

            if auto_mode and position_accuracy_list:
                self.align_bar.setValue(int(position_accuracy_list[-1]))
                self.rotation_bar.setValue(int(rotation_accuracy_list[-1]) if rotation_accuracy_list else 0)
            else:
                self.align_bar.setValue(0)
                self.rotation_bar.setValue(0)

        if auto_mode:
            #self.label_message.setText("AUTO")
            self.position_completed_shown = False
            self.rotation_completed_shown = False
        elif align_done:
            if not self.position_completed_shown:
                #self.label_message.setText("POSITION completed")
                self.position_completed_shown = True
                self.message_timer.start(2000)
        elif rotation_done:
            if not self.rotation_completed_shown:
                #self.label_message.setText("ROTATION completed")
                self.rotation_completed_shown = True
                self.message_timer.start(2000)
        elif send_enabled:
            #self.label_message.setText(self.t("Positioning..."))
            self.position_completed_shown = False
            self.rotation_completed_shown = False
        elif self.rotation_active:
            #self.label_message.setText(self.t("Rotationing..."))
            self.position_completed_shown = False
            self.rotation_completed_shown = False
        elif home_mode:
            #self.label_message.setText(self.t("Defaulting..."))
            self.position_completed_shown = False
            self.rotation_completed_shown = False
        else:
            #self.label_message.setText(self.t("Standby"))
            self.position_completed_shown = False
            self.rotation_completed_shown = False
        # 12) 정렬 모드 로직
        if confirmed_center:
            dx, dy, dx_s, dy_s, cmd = calculate_steps(confirmed_center, Target_Point)
            self.label_coords.setText(self.t("Position_Err") + f":\n({dx}, {dy})")
            if send_enabled:
                acc = calculate_accuracy_px(Target_Point, confirmed_center)
                self.align_bar.setValue(int(acc))
            if abs(dx)<=TOLERANCE_PX and abs(dy)<=TOLERANCE_PX:
                stable_count +=1
                if stable_count>=STABLE_REQUIRED:
                    align_done=True; send_enabled=False; stable_count=0
                    #self.label_message.setText(" POSITION completed ")
                    asyncio.create_task(self.send_serial_command("STOP_ALIGN"))
            else:
                stable_count=0
                if (send_enabled and stable_count<STABLE_REQUIRED
                    and not awaiting_done and not sending_command
                    and cmd and (cmd!=last_sent_command or force_send)):
                    asyncio.create_task(self.send_serial_command(cmd))
                    force_send=False
        elif auto_mode:
            self.align_bar.setValue(int(position_accuracy_list[-1]) if position_accuracy_list else 0)
            self.rotation_bar.setValue(int(rotation_accuracy_list[-1]) if rotation_accuracy_list else 0)
        else:
            self.label_coords.setText(self.t("Position_Err") + ": -")
            if not (align_done or rotation_done or auto_mode):
                self.reset_progress()

        if confirmed_center:
            cx, cy = confirmed_center
            cv2.circle(display, (cx,cy), 5, (0,255,0), -1)
            cv2.putText(display, f"({cx},{cy})",
                        (cx+10,cy+10), FONT, FONT_SCALE, (0,255,0), FONT_THICKNESS)
            cv2.line(display, tuple(Target_Point), (cx,cy), (255,0,0), 2)

        cv2.circle(display, tuple(Target_Point), 5, (0,0,255), -1)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(img))

        half = 50
        x1 = max(0, Target_Point[0]-half)
        y1 = max(0, Target_Point[1]-half)
        x2 = min(display.shape[1], Target_Point[0]+half)
        y2 = min(display.shape[0], Target_Point[1]+half)
        zoom = display[y1:y2, x1:x2]
        if zoom.size>0 and hasattr(self, 'zoom_window') and self.zoom_window:
            h2,w2,ch2 = zoom.shape
            qimg = QImage(zoom.data.tobytes(), w2, h2, ch2*w2, QImage.Format_RGB888)
            self.zoom_window.zoom_label.setPixmap(QPixmap.fromImage(qimg))

        if (send_enabled and stable_count<STABLE_REQUIRED
            and not awaiting_done and not sending_command
            and cmd and cmd!=last_sent_command and not force_send):
            asyncio.create_task(self.send_serial_command(cmd))

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
        self.setWindowTitle("Log")
        self.setGeometry(200, 200, 600, 400)
        self.logs = []
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.filter_box = QComboBox()
        self.filter_box.addItems(["ALL", "ALIGN", "ROTATION", "CHANGED", "ERROR", "Status"])
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
        self.setWindowTitle(self.t("Alignment_Zoom"))
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
