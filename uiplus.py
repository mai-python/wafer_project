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
#openCV 영상 해상도(가로,세로)
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
#정렬 목표 좌표(기본은 화면 중앙값)
Target_Point = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]
#원위치 목표 좌표
HOME_DXDY = (400, 200)
#오차범위 값 1픽셀, 1도
TOLERANCE_PX, TOLERANCE_R = 1, 1
#1픽셀 당 몇 mm인지 변환
PIXEL_TO_MM_X, PIXEL_TO_MM_Y = 0.6, 0.83
#1스텝 당 갈 수 있는 mm, 최대 스텝수 제한
STEPS_PER_MM, MAX_STEPS = 55, 32000
# 컴퓨터 내 카메라 번호, 노트북은 0번이 웹캠임
CAMERA_INDEX = 0
#아두이노 시리얼 통신 속도
BAUDRATE = 9600
#아두이노 시리얼 포트 
SERIAL_PORT = "COM6"
#웨이퍼(노란색 원) 인식을 위한 HSV 범위
YELLOW_LOWER, YELLOW_UPPER = (15, 100, 100), (40, 255, 255)
#객체 인식 범위 및 원형 범위 조건
AREA_THRESHOLD, ROUNDNESS_THRESHOLD = 1000, 0.4
#만족해야 루프종료(정렬완료 등)
STABLE_REQUIRED = 3
#화면 출력용 글꼴
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE, FONT_THICKNESS = 0.6, 2
TEXT_COLOR = (255, 255, 255)
#로테이션 변수: 목표각도 및 현재각도
Target_degree, angle = 0, 0
#각도를 스텝으로 변환, 최대 회전스텝 제한
DEGREE_TO_STEP, MAX_ROTATION_STEP = 1, 32000
#화면에 보이는 바운딩 박스(클래스) 색 조정
CLASS_COLORS = {
    'F1': (0, 0, 50), 'F2': (0, 0, 100),
    'P100': (0, 0, 255), 'P111': (255, 0, 0),
}
#detection_summary에 사용되는 그래프 구성성분
detection_stats = {
    "P100": {"F1": [], "F2": [], "P100": [], "P111": []},
    "P111": {"F1": [], "F2": [], "P100": [], "P111": []}
}
##전역변수##
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
    # 경고 메시지 보내는 함수, title: 화면(창) 제목 message: 메시지 내용
    warning.setStandardButtons(QMessageBox.Ok)
    warning.exec_()

def calculate_accuracy_px(target_point, center_point):
    #중심 좌표 정렬 accuracy(정확도) 구하는 함수
    dx = target_point[0] - center_point[0]
    dy = target_point[1] - center_point[1]
    error_distance = (dx ** 2 + dy ** 2) ** 0.5
    #두 point 사이의 거리로 accuracy계산[%단위] 
    MAX_ERROR_PX = 30        
    PERFECT_THRESH_PX = 1.5   

    if error_distance <= PERFECT_THRESH_PX: #error_distance: Target_point와 center_point사이의 거리
        #perfect_thresh_px보다 작거나 같다면 정확도:100%
        return 100.0
    #아니면 정확도0%, 30px이내면 linear하게 정확도 상승
    elif error_distance >= MAX_ERROR_PX:
        return 0.0 
    else:
        # 정확도 계산 공식
        return round((1 - (error_distance / MAX_ERROR_PX)) * 100, 2)
    
def log(msg, window):
    # 현재 시각을 시간,분,초로 나타냄
    timestamp = datetime.now().strftime("[%H:%M:%S]")
    # 로그 msg 생성: [시간:분:초, 내용]
    full_msg = f"{timestamp} {msg}" 
    #로그를 터미널에 출력
    print(full_msg)
    #log_window(로그창) 있으면 거기에도 같은 메시지 출력, 총 2개의 로그가 출력됨
    if hasattr(window, 'log_window') and window.log_window:
        window.log_window.append_log(full_msg)

def compute_roundness(cnt):
    # cv가 계산한 contour의 면적 계산
    area = cv2.contourArea(cnt)
    #contour 둘레 계산, True는 곡선          #contour: 물체의 윤곽선(엣지)
    perimeter = cv2.arcLength(cnt, True)
    #둘레가 0이면, roundness는 0으로 > 계산 무효
    if perimeter == 0:
        return 0
    #roundness(원형도) 계산 공식, 1에 가까울수록 좋음
    return 4 * np.pi * area / (perimeter * perimeter)

def detect_best_yellow_circle(frame):
    #hsv 형식으로 바꾸어 노란색 범위를 추출(웨이퍼는 노란색이니까)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #노란색 범위 설정(어디까지가 노란색인지)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    #gaussianblur(가우시안 블러): 흐리게 처리하여 경계선의 노이즈를 제거 및 mask 인식 정확도 향상
    mask = cv2.GaussianBlur(mask, (7, 7), 2)
    #mask에서 contour추출하여 외곽만 사용
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #best를 찾기 위한 초기 조건(얼마나 둥근지, 면적이 적당한지)
    best = None  #초기화
    best_score = 0 #초기화
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_THRESHOLD:
            continue
        #roundness가 기준보다 크거나 같으면 인정
        roundness = compute_roundness(cnt)
        if roundness >= ROUNDNESS_THRESHOLD:
            score = area * roundness
            (x, y), _ = cv2.minEnclosingCircle(cnt)
            #만약 현재 스코어가 최고점보다 크면 갱신
            if score > best_score:
                best = (int(x), int(y))
                best_score = score
    #best값 반환
    return best

def draw_yolo_boxes(frame, results, window=None, override_type=None):
    #YOLO가 탐지 후 boxes.data를 찾아봄
    for box in results[0].boxes.data:
        #x1,y2: 바운딩 박스 왼쪽 위 좌표값, x2,y2: 바운딩 박스 오른쪽 아래 좌표
        #conf: 신뢰도(정확도), ex) F1: conf = 0.7이면 실제로 F1, conf=.2면 오탐지 가능성 높음
        x1, y1, x2, y2, conf, cls = box.tolist()
        #클래스(F1, F2, 라벨링이라 생각하면 됨)
        class_name = results[0].names[int(cls)]
        # F1, F2아니면 패스
        if class_name not in ["F1", "F2"]:
            continue
        #바운딩 박스 및 텍스트 꾸미기
        color = CLASS_COLORS.get(class_name, (0, 255, 0))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_name}"
        ((tw, th), _) = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(frame, (int(x1), int(y1 - th - 4)), (int(x1 + tw), int(y1)), color, -1)
        cv2.putText(frame, label, (int(x1), int(y1 - 4)), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

def detect_yolo_center(frame):
    #openCv는 BGR로 YOLO는 RGB로 바꾸어야함
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #결과값은 results로
    results = model(rgb)
    #YOLO가 탐지한 바운딩 박스의 정보(좌표,conf,class)를 처리
    for box in results[0].boxes.data:
        x1, y1, x2, y2, *_ = box.tolist()

        #보정값
        cx = int((x1 + x2) / 2)
        cy = FRAME_HEIGHT - int((y1 + y2) / 2)
        return (cx, cy)
    #박스없으면 None반환
    return None

def calculate_angle(pt1, pt2):
    # 두 점 사이의 기울기로 각도 계산 함수
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    # 각도는 아크탄젠트로 구하고, radian을 degree로 변환
    angle = np.degrees(np.arctan2(dy, dx))
    #각도에 360을 더해서 (+)값으로 보정
    return angle + 360 if angle < 0 else angle

def calculate_rotation(current_angle, target_angle):
    #두 angle의 차이를 -180~180으로 규정. 구하는 이유: rotation할 때 가장 빠른 방향으로 가기 위해
    #그러나 우리 코드에서는 diR = 1(rotation방향)을 하나로 정의했기 때문에 의미없음
    dR = (target_angle - current_angle + 540) % 360 - 180
    diR = 1
    stR = max(1, min(int(abs(dR) * DEGREE_TO_STEP), MAX_ROTATION_STEP))
    return diR, stR

def calculate_steps(from_point, to_point):
    dx = to_point[0] - from_point[0] #x축 이동거리
    dy = from_point[1] - to_point[1] #y축 이동거리
    dx_mm = abs(dx) * PIXEL_TO_MM_X #x축 이동거리를 mm단위 변환
    dy_mm = abs(dy) * PIXEL_TO_MM_Y #y축 이동거리를 mm단위 변환
    dx_steps = min(int(dx_mm * STEPS_PER_MM), MAX_STEPS) # x축 스텝 수로 변환
    dy_steps = min(int(dy_mm * STEPS_PER_MM), MAX_STEPS) # y축 스텝 수로 변환
    dix = 1 if dx > 0 else 0 #x 방향 좌우 설정
    diy = 1 if dy > 0 else 0 #y 방향 좌우 설정
    command = f"{dix},{diy},{dx_steps},{dy_steps},0,0" # 아두이노에게 보낼 명령 문자열 생성 이때 0,0은 회전명령임
    return dx, dy, dx_steps, dy_steps, command
##########################################수정금지#########################################################
class MainWindow(QMainWindow):
    def __init__(self, loop): 
        super().__init__() # 초기화
        self.loop = loop
        # 프로그램 창 제목
        self.setWindowTitle("Wafer_Aligner")
        self.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT + 200 + self.menuBar().height())
        self.setMouseTracking(True)
        #메뉴바 생성
        menubar = self.menuBar()
        #FILE 메뉴
        file_menu = menubar.addMenu("FILE")
        file_menu.addAction(QAction("EXIT", self, triggered=self.close)) # 프로그램 종료
        capture_action = QAction("Capture", self) # 프로그램 화면 이미지 캡쳐
        capture_action.triggered.connect(self.capture_screenshot)
        file_menu.addAction(capture_action)
        #Function 메뉴
        function_menu = menubar.addMenu("Function")
        function_menu.addAction(QAction("Align", self, triggered=self.show_align_tab)) # Align 탭 생성 기능
        function_menu.addAction(QAction("Rotation", self, triggered=self.show_angle_tab)) # Rotation 탭 생성 기능
        function_menu.addAction(QAction("Advanced", self, triggered=self.show_advanced_tab)) # Advanced 탭 생성 기능
        #Tool 메뉴
        tool_menu = menubar.addMenu("Tool")
        settings_menu = QMenu("Settings", self) # 변수 및 그래프 보기 기능
        tool_menu.addAction(QAction("RELOAD_MODE_YOLO", self, triggered=self.reload_yolo_model)) # 모델 다시불러오기, 상태 좋아지게하는 기능
        tool_menu.addMenu(settings_menu)    
        settings_menu.addAction("Show Graph", self.show_graph_tab) # 그래프 보기
        settings_menu.addAction("Variables", self.show_variable_tab) # 변수 설정하기
        #help 메뉴
        help_menu = menubar.addMenu("Help") 
        help_menu.addAction(QAction("Guide", self, triggered=self.show_help_dialog)) # 도움말
        #프로그램 디스플레이 출력
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        #화면크기
        self.cap.set(3, FRAME_WIDTH) 
        self.cap.set(4, FRAME_HEIGHT)

        self.zoom_window = None # F2누르면 나오는 확대창
        self.log_window = LogWindow() # F3누르면 나오는 로그창
        self.rotation_active = False # 회전 정렬 변수
        self.warned_once = False # 카메라 경고 변수
        #그래프 탭
        self.graph_tab = QWidget()
        graph_layout = QVBoxLayout()
        self.graph_canvas = GraphCanvas(self)
        graph_layout.addWidget(self.graph_canvas)
        self.graph_tab.setLayout(graph_layout)
        #디스플레이 라벨
        self.image_label = QLabel()
        self.image_label.setScaledContents(True)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT)
        #탭 ui
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.hide() # 처음에 숨기고
        self.setup_tabs() # 탭 누적? 쌓이게
        # 가운데 ui (동작 상태 및 진행률 막대)
        self.label_message = QLabel(" 대기중 ")
        self.label_message.setStyleSheet("font-size: 16px; color: black;")
        self.align_label = QLabel("ALIGN_PROGRESS") # 진행률 막대
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
        # 오른쪽 ui (status)
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
        # 왼쪽 패널(탭과 버튼 포함)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.tabs)
        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(520)
        #프로그램아래여백
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)
        bottom_layout.addWidget(left_panel)
        bottom_layout.addWidget(self.progress_widget)
        bottom_layout.addWidget(self.status_widget)
        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_layout)
        bottom_widget.setFixedHeight(200)
        #메인화면 레이아웃
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(bottom_widget)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        # progress바에 사용되는 변수
        self.initial_offset = (100, 100)
        self.initial_angle = 140
        # 아두이노 관련 
        self.ser_reader = None
        self.ser_writer = None
        self.serial_ready = False
        # 정렬 대기용 타이머(3초뒤 움직이게 하는 auto_align에서 사용)
        self.reset_timer = QTimer()
        self.reset_timer.setSingleShot(True)
        self.reset_timer.timeout.connect(self.set_waiting)
        # update_frame에서 사용하는 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.setMouseTracking(True)
        self.show() # 화면 표시

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
        #decimals: 소수점n자리까지 min/max: 범위 제한
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
        
        is_critical_stop = command.upper() == "STOP" or command == "STOP_ALIGNMENT"
    
        if (sending_command or awaiting_done) and not is_critical_stop:
            return
        if command == last_sent_command and not force_send and not is_critical_stop:
            return
    
        is_stop_alignment_command = (command == "STOP_ALIGNMENT")
    
        sending_command = True
        awaiting_done = True
        try:
            actual_command_to_send = command
            if command.upper() == "STOP":
                actual_command_to_send = "0,0,0,0,0,0"
            elif is_stop_alignment_command:
                actual_command_to_send = "0,0,0,0,0,0"
            
            current_label_state = self.label_state.text()
    
            if home_mode:
                self.label_state.setText("Status: HOMING")
                # self.label_message.setText("원위치 중")
            elif is_stop_alignment_command:
                self.label_state.setText("Status: ALIGN_STOPPING")
                # self.label_message.setText("정렬 중단 중") 
            elif actual_command_to_send == "0,0,0,0,0,0": # 일반 STOP
                self.label_state.setText("Status: STOPPING") 
                # self.label_message.setText("작업 중지 중") 
            elif "0,0,0,0,1," in actual_command_to_send or "0,0,0,0,0,1" in actual_command_to_send: 
                self.label_state.setText("Status: ROTATIONING") 
                # self.label_message.setText("회전 중...") 
            else: # 일반 정렬 단계
                self.label_state.setText("Status: ALIGNING") 
                # self.label_message.setText("정렬 단계 진행 중...") 
    
    
            self.ser_writer.write((actual_command_to_send + "\n").encode())
            await self.ser_writer.drain()
            last_sent_command = actual_command_to_send
            force_send = False
    
            while True:
                try:
                    line = await asyncio.wait_for(self.ser_reader.readline(), timeout=30.0)
                    decoded = line.decode().strip()
                    if "DONE" in decoded or "READY" in decoded:
                        break
                except asyncio.TimeoutError:
                    if home_mode: self.label_state.setText("Status: HOMING_TIMEOUT")
                    elif is_stop_alignment_command : self.label_state.setText("Status: ALIGN_STOP_TIMEOUT")
                    elif "0,0,0,0,1," in actual_command_to_send or "0,0,0,0,0,1" in actual_command_to_send : self.label_state.setText("Status: ROTATION_TIMEOUT")
                    else: self.label_state.setText("Status: CMD_TIMEOUT")
                    break
        except Exception as e:
            log(f"[ERR] Serial communication error: {e}", self) 
            self.label_state.setText("Status: SERIAL_ERROR") 
            # self.label_message.setText("시리얼 통신 오류") 
            pass 
        finally:
            sending_command = False
            awaiting_done = False
            if home_mode:
                self.label_state.setText("Status: HOMING_DONE")
                # self.label_message.setText("원위치 완료") 
                home_mode = False
                send_enabled = False
            elif is_stop_alignment_command:
                self.label_state.setText("Status: Standby") 
                self.label_message.setText(" 대기중 ")      
            else:
                if last_sent_command == "0,0,0,0,0,0": # 일반 "STOP" 완료 (회전 종료, 자동모드 종료 등)
                    self.label_state.setText("Status: CMD_STOPPED")
                    self.reset_timer.start(1000)
                else: 
                    current_op_description = "명령"
                    if "0,0,0,0,1," in last_sent_command or "0,0,0,0,0,1" in last_sent_command:
                        self.label_state.setText("Status: ROTATION_STEP_DONE")
                    else:
                        self.label_state.setText("Status: ALIGN_STEP_DONE")
                    self.reset_timer.start(2000) 

def toggle_send(self):
    global send_enabled, stable_count, Target_Point 
    send_enabled = not send_enabled

    if send_enabled:
        stable_count = 0
        self.warned_once = False
        if confirmed_center: 
            initial_dx = Target_Point[0] - confirmed_center[0]
            initial_dy = Target_Point[1] - confirmed_center[1]
            self.initial_offset = (abs(initial_dx), abs(initial_dy))
            else
                self.initial_offset = (FRAME_WIDTH, FRAME_HEIGHT)
        
        self.label_state.setText("Status: ALIGNING")
        self.label_message.setText("정렬 중...")
    else:
        self.label_state.setText("Status: Standby") 
        self.label_message.setText(" 대기중 ")
        asyncio.create_task(self.send_serial_command("STOP_ALIGNMENT"))

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
        global confirmed_center, stable_count, angle, send_enabled, Target_Point, \
               home_mode, awaiting_done, sending_command, \
               has_f1, has_f2, Target_degree 


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
            # elif label == "F2":
            #     f2_center = (cx, cy)#F2만 찾는건 의미없으니 빼놓음
            
            if wafer_type in ["P100", "P111"]:
                for inner_box in results[0].boxes.data: 
                    _, _, _, _, inner_conf, inner_cls = inner_box.tolist() 
                    inner_label = results[0].names[int(inner_cls)] 
                    if inner_label in detection_stats[wafer_type]:
                        detection_stats[wafer_type][inner_label].append(inner_conf)

        if f1_center and wafer_center:
            current_angle_val = calculate_angle(f1_center, wafer_center) 
            angle = current_angle_val 
            angle_text = f"{current_angle_val:.2f}°"
            self.label_angle.setText(f"Angle: {current_angle_val:.2f}°")
    
            diff = (Target_degree - current_angle_val + 540) % 360 - 180
            angle_error_val = abs(diff)
            self.label_angle_error.setText(f"Angle_Err: {angle_error_val:.2f}°") 
            cv2.putText(display, angle_text, (f1_center[0] + 10, f1_center[1] - 10), FONT, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)
        else:
            angle_error_val = self.initial_angle 
    
        center_from_yellow = detect_best_yellow_circle(display)
        if center_from_yellow:
            confirmed_center = center_from_yellow
        elif wafer_center: # YOLO가 P100/P111을 감지했다면 그 중심 사용
            confirmed_center = wafer_center
        else:
            confirmed_center = None
    
        if confirmed_center:
            dx, dy, dx_steps, dy_steps, command = calculate_steps(confirmed_center, Target_Point)
            self.label_coords.setText(f"Offset: ({dx}, {dy})")
            self.update_progress(dx, dy, angle_error_val if 'angle_error_val' in locals() else self.initial_angle)
    
    
            if abs(dx) <= TOLERANCE_PX and abs(dy) <= TOLERANCE_PX:
                if send_enabled and not awaiting_done and not sending_command:
                    stable_count += 1
                    if stable_count >= STABLE_REQUIRED:
                        # global send_enabled # 이미 함수 상단에 선언됨
                        send_enabled = False
                        stable_count = 0
                        self.label_state.setText("Status: Standby") # 최종 목표 상태
                        self.label_message.setText(" 정렬 완료 ")
                        asyncio.create_task(self.send_serial_command("STOP_ALIGNMENT"))
                        return 
            else:
                stable_count = 0
    
            if send_enabled and stable_count < STABLE_REQUIRED:
                asyncio.create_task(self.send_serial_command(command))
            
            accuracy = calculate_accuracy_px(Target_Point, confirmed_center)
            self.graph_canvas.update_plot(accuracy) # 그래프 업데이트
            self.label_accuracy.setText(f"Accuracy: {accuracy}%")
        else:
            # confirmed_center를 찾지 못한 경우
            self.label_coords.setText("Offset: (N/A)")
            self.label_accuracy.setText("Accuracy: 0%")
            # dx, dy가 없으므로 프로그레스바는 최대 오차로 표시하거나 이전 값 유지
            self.update_progress(self.initial_offset[0], self.initial_offset[1], self.initial_angle if 'angle_error_val' not in locals() else angle_error_val)

        

        if not self.rotation_active:  
            cv2.circle(display, tuple(Target_Point), 5, (0, 0, 255), -1)
            if confirmed_center:
                cv2.circle(display, confirmed_center, 5, (0, 255, 0), -1)
                cv2.line(display, tuple(Target_Point), confirmed_center, (255, 0, 0), 2)

        self.label_wafer_type.setText(f"Wafer Type: {wafer_type if wafer_type else '-'}")

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
