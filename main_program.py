import cv2
import numpy as np
import time
import serial
from ultralytics import YOLO

# ========================== 설정값 ==========================

FRAME_WIDTH = 1280  # 영상 프레임 가로 크기 (픽셀 단위)
FRAME_HEIGHT = 720  # 영상 프레임 세로 크기

Target_Point = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]  # 기준점: 화면 중앙부터 시작 (마우스로 수정 가능)
TRACK_CONFIRM_TIME = 2.0  # 객체 중심이 일정 시간 유지되어야 확정됨 (초 단위)

HOME_DXDY = (100, 100)

PIXEL_TO_MM_X = 0.6     # X축 픽셀당 mm 변환 비율
PIXEL_TO_MM_Y = 0.83    # Y축 픽셀당 mm 변환 비율
STEPS_PER_MM = 1600     # 스텝모터가 1mm 이동하는 데 필요한 스텝 수
TOLERANCE_PX = 10       # 기준점과 객체 중심 간 허용 오차 (픽셀 단위)
MAX_STEPS = 32000       # 한 번에 보낼 수 있는 최대 스텝 수

SERIAL_PORT = ""        # 아두이노 연결 포트 (예: "/dev/ttyUSB0")
CAMERA_INDEX = 4        # 사용할 카메라 인덱스
BAUDRATE = 9600         # 아두이노와의 통신 속도

X_RANGE = (-200, 200)   # 마우스 클릭으로 이동 가능한 범위 (X축 기준)
Y_RANGE = (-200, 200)   # 마우스 클릭으로 이동 가능한 범위 (Y축 기준)
HOME_OFFSET = (0, 0)    # 정렬 후 복귀할 기준 위치 (스텝 단위)

MARGIN = 100            # 객체가 프레임 가장자리에서 몇 픽셀 이내일 때 '가장자리'로 간주
AREA_THRESHOLD = 1000   # contour 최소 면적 – 노이즈 제거용
YELLOW_LOWER = (20, 100, 100)  # HSV 기준: 노란색 하한값
YELLOW_UPPER = (30, 255, 255)  # HSV 기준: 노란색 상한값

# ========================== 상태 변수 ==========================

log_buffer = []
last_detected = None
confirm_start_time = None
confirmed_center = None
already_moved = False
tracking_stopped = False
show_limit_box = False

dx = dy = 0
dix = diy = 0
dx_steps = dy_steps = 0
command = ""

# ========================== YOLO 로드 ==========================

model = YOLO("yolov8n.pt")

# ========================== 함수 정의 ==========================

def is_near_edge(cx, cy, margin=MARGIN):
    # 가장자리 여부 판단 (영상 외곽에 가까운지 확인)
    return (
        cx < margin or
        cx > FRAME_WIDTH - margin or
        cy < margin or
        cy > FRAME_HEIGHT - margin
    )

def compute_center(cnt):
    # 컨투어 중심 계산
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        return (int(ellipse[0][0]), int(ellipse[0][1]))
    else:
        (x, y), _ = cv2.minEnclosingCircle(cnt)
        return (int(x), int(y))

def detect_opencv_priority_center(frame):
    # 노란색 물체 탐지 – 가장자리 우선
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edge_center = None
    normal_center = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_THRESHOLD:
            continue
        cx, cy = compute_center(cnt)
        if is_near_edge(cx, cy):
            edge_center = (cx, cy)
            break
        elif not normal_center:
            normal_center = (cx, cy)

    return edge_center if edge_center else normal_center

def log(msg):
    # 로그 출력 및 저장
    if len(log_buffer) == 0 or log_buffer[-1] != msg:
        print(msg)
        log_buffer.append(msg)
        if len(log_buffer) > 15:
            log_buffer.pop(0)

def draw_limit_box(frame):
    # 이동 제한 영역 사각형 그리기
    x1 = Target_Point[0] + X_RANGE[0]
    y1 = FRAME_HEIGHT - (Target_Point[1] + Y_RANGE[1])
    x2 = Target_Point[0] + X_RANGE[1]
    y2 = FRAME_HEIGHT - (Target_Point[1] + Y_RANGE[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

def mouse_callback(event, x, y, flags, param):
    # 마우스 클릭으로 Target_Point 설정
    if event == cv2.EVENT_LBUTTONDOWN:
        tx = x
        ty = FRAME_HEIGHT - y
        dx_check = tx - Target_Point[0]
        dy_check = ty - Target_Point[1]
        if X_RANGE[0] <= dx_check <= X_RANGE[1] and Y_RANGE[0] <= dy_check <= Y_RANGE[1]:
            Target_Point[0] = tx
            Target_Point[1] = ty
            log(f"[CLICK] Target_Point set to ({tx}, {ty})")
        else:
            log("[DENIED] 클릭 범위 초과. 무시됨.")

def show_log_window():
    # 로그 정보 OpenCV 창에 표시
    log_img = np.ones((400, 700, 3), np.uint8) * 30
    y = 25
    cv2.putText(log_img, f"Target_Point:      {tuple(Target_Point)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1); y += 22
    cv2.putText(log_img, f"last_detected:     {last_detected if last_detected else '(None)'}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1); y += 22
    cv2.putText(log_img, f"confirmed_center:  {confirmed_center if confirmed_center else '(None)'}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1); y += 22
    if confirmed_center:
        cv2.putText(log_img, f"dx = {dx}, dy = {-dy}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1); y += 22
        cv2.putText(log_img, f"Xdir={dix} Ydir={diy} Xstep={dx_steps} Ystep={dy_steps}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1); y += 22
        cv2.putText(log_img, f"Serial Command: {command}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1); y += 30
    else:
        y += 66
    for line in log_buffer[-10:]:
        cv2.putText(log_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        y += 20
    cv2.imshow("Log Window", log_img)

def move_stage(dx_pix, dy_pix):
    # 모터 이동 명령 계산 및 전송
    global dx, dy, dix, diy, dx_steps, dy_steps, command
    dx = dx_pix
    dy = dy_pix
    dix = 1 if dx >= 0 else 0
    diy = 1 if dy >= 0 else 0
    dx_mm = abs(dx) * PIXEL_TO_MM_X
    dy_mm = abs(dy) * PIXEL_TO_MM_Y
    dx_steps = min(int(dx_mm * STEPS_PER_MM), MAX_STEPS)
    dy_steps = min(int(dy_mm * STEPS_PER_MM), MAX_STEPS)
    command = f"{dix},{diy},{dx_steps},{dy_steps}"
    try:
        with serial.Serial(SERIAL_PORT, BAUDRATE, timeout=3) as ser:
            time.sleep(2)
            ser.flushInput()
            ser.write((command + "\n").encode())
            time.sleep(0.3)
            start = time.time()
            while time.time() - start < 3:
                if ser.in_waiting:
                    _ = ser.readline().decode(errors='ignore').strip()
    except Exception as e:
        log(f"[ERR] Serial: {e}")

def detect_yolo_center(frame):
    # YOLO로 객체 중심 추출
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb)
    for box in results[0].boxes.data:
        x1, y1, x2, y2, _, _ = box.tolist()
        cx = int((x1 + x2) / 2)
        cy = FRAME_HEIGHT - int((y1 + y2) / 2)
        return ((cx, cy), 0)
    return [(tuple(Target_Point))]

def is_close(p1, p2, tolerance=TOLERANCE_PX):
    # 중심이 목표점에 근접했는지 판단
    return abs(p1[0] - p2[0]) <= tolerance and abs(p1[1] - p2[1]) <= tolerance

def return_to_origin():
    # HOME_OFFSET으로 복귀
    dx_home = HOME_OFFSET[0]
    dy_home = HOME_OFFSET[1]
    move_stage(dx_home, dy_home)
    log("[Return] 원위치로 이동 완료.")

def return_to_home_dxdy():
    global confirmed_center
    if confirmed_center is None:
        log("[HOME] 현재 위치 정보 없음, 이동 생략.")
        return
    dx = HOME_DXDY[0] - confirmed_center[0]
    dy = HOME_DXDY[1] - confirmed_center[1]
    move_stage(dx,dy)
    log(f"[HOME] 원위치 {HOME_DXDY}로 원위치 완료.")

# ========================== 메인 루프 ==========================

def main():
    global last_detected, confirm_start_time, confirmed_center
    global already_moved, tracking_stopped, show_limit_box

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)

    cv2.namedWindow("Wafer Align")
    cv2.setMouseCallback("Wafer Align", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            log("[Camera] Read Fail")
            break

        if tracking_stopped:
            show_log_window()
            if show_limit_box:
                draw_limit_box(frame)
            cv2.imshow("Wafer Align", frame)
            if cv2.waitKey(1) & 0xFF == ord('r'):
                log("[RESET] Tracking re-enabled")
                confirmed_center = None
                already_moved = False
                tracking_stopped = False
                last_detected = None
                confirm_start_time = None
                log_buffer.clear()
            continue

        wafer = detect_opencv_priority_center(frame)
        if not wafer:
            wafer = detect_yolo_center(frame)[0]

        if wafer:
            if last_detected and is_close(wafer, last_detected):
                if confirm_start_time and time.time() - confirm_start_time > TRACK_CONFIRM_TIME and not confirmed_center:
                    confirmed_center = wafer
            else:
                confirm_start_time = time.time()
                last_detected = wafer

        if confirmed_center and not already_moved:
            dx_local = confirmed_center[0] - Target_Point[0]
            dy_local = Target_Point[1] - confirmed_center[1]

            if not (X_RANGE[0] <= dx_local <= X_RANGE[1] and Y_RANGE[0] <= dy_local <= Y_RANGE[1]):
                log("[BLOCKED] 이동 범위 초과. 이동 취소됨.")
                tracking_stopped = True
                already_moved = True
            else:
                move_stage(dx_local, dy_local)
                time.sleep(1)
                return_to_origin()
                already_moved = True
                tracking_stopped = True

        if wafer:
            cv2.circle(frame, wafer, 5, (0,255,0), -1)
            cv2.circle(frame, tuple(Target_Point), 5, (0,0,255), -1)
            cv2.line(frame, wafer, tuple(Target_Point), (255,0,0), 2)

        if show_limit_box:
            draw_limit_box(frame)

        show_log_window()
        cv2.imshow("Wafer Align", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            log("[RESET] Manual reset")
            last_detected = None
            confirm_start_time = None
            confirmed_center = None
            already_moved = False
            tracking_stopped = False
            log_buffer.clear()
        elif key == ord('w'):
            show_limit_box = not show_limit_box
        elif key == ord('u'):
            return_to_origin()
            log("[Manual] Returned to origin")
        continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
