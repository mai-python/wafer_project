import cv2
import numpy as np
import time
import serial
from ultralytics import YOLO
import threading

# ========================== 설정값 ==========================
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
Target_Point = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]
TRACK_CONFIRM_TIME = 1.0

HOME_DXDY = (300, 100)

PIXEL_TO_MM_X = 0.6
PIXEL_TO_MM_Y = 0.83
STEPS_PER_MM = 55
TOLERANCE_PX = 5
MAX_STEPS = 32000
MAX_ALIGNMENT_ATTEMPTS = 5  # 오차 범위 도달 시까지 최대 반복 횟수

SERIAL_PORT = "/dev/ttyACM1"
CAMERA_INDEX = 0
BAUDRATE = 9600

X_RANGE = (-1000, 1000)
Y_RANGE = (-600, 600)
HOME_OFFSET = (0, 0)

MARGIN = 100
AREA_THRESHOLD = 1000
YELLOW_LOWER = (20, 100, 100)
YELLOW_UPPER = (30, 255, 255)

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

model = YOLO("yolov8n.pt")

def is_near_edge(cx, cy, margin=MARGIN):
    return cx < margin or cx > FRAME_WIDTH - margin or cy < margin or cy > FRAME_HEIGHT - margin

def compute_center(cnt):
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        return (int(ellipse[0][0]), int(ellipse[0][1]))
    else:
        (x, y), _ = cv2.minEnclosingCircle(cnt)
        return (int(x), int(y))

def detect_opencv_priority_center(frame):
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
    if len(log_buffer) == 0 or log_buffer[-1] != msg:
        print(msg)
        log_buffer.append(msg)
        if len(log_buffer) > 15:
            log_buffer.pop(0)

def draw_limit_box(frame):
    x1 = Target_Point[0] + X_RANGE[0]
    y1 = FRAME_HEIGHT - (Target_Point[1] + Y_RANGE[1])
    x2 = Target_Point[0] + X_RANGE[1]
    y2 = FRAME_HEIGHT - (Target_Point[1] + Y_RANGE[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

def mouse_callback(event, x, y, flags, param):
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
    if dx_pix == 0 and dy_pix == 0:
        log("[SKIP] move_stage called with (0, 0) → Skipping motor move.")
        return

    global dx, dy, dix, diy, dx_steps, dy_steps, command
    dx = dx_pix
    dy = dy_pix
    dix = 1 if dx <= 0 else 0
    diy = 1 if dy <= 0 else 0
    dx_mm = abs(dx) * PIXEL_TO_MM_X
    dy_mm = abs(dy) * PIXEL_TO_MM_Y
    dx_steps = min(int(dx_mm * STEPS_PER_MM), MAX_STEPS)
    dy_steps = min(int(dy_mm * STEPS_PER_MM), MAX_STEPS)
    command = f"{dix},{diy},{dx_steps},{dy_steps}"
    log(f"[DEBUG] dx_steps = {dx_steps}, dy_steps = {dy_steps}")

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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb)
    for box in results[0].boxes.data:
        x1, y1, x2, y2, _, _ = box.tolist()
        cx = int((x1 + x2) / 2)
        cy = FRAME_HEIGHT - int((y1 + y2) / 2)
        return ((cx, cy), 0)
    return [(tuple(Target_Point))]

def is_close(p1, p2, tolerance=TOLERANCE_PX):
    return abs(p1[0] - p2[0]) <= tolerance and abs(p1[1] - p2[1]) <= tolerance

def return_to_origin():
    dx_home = HOME_OFFSET[0]
    dy_home = HOME_OFFSET[1]
    move_stage(dx_home, dy_home)
    log("[Return] move complete.")

def return_to_home_dxdy():
    global confirmed_center
    if confirmed_center is None:
        log("[HOME] move pass.")
        return
    dx = HOME_DXDY[0] - confirmed_center[0]
    dy = HOME_DXDY[1] - confirmed_center[1]
    move_stage(dx, dy)
    log(f"[HOME] 원위치 {HOME_DXDY}로 원위치 완료.")
alignment_thread = None
alignment_done = False

def threaded_alignment():
    global already_moved, tracking_stopped, alignment_done, alignment_thread
    wafer = detect_opencv_priority_center(frame_global)
    if not wafer:
        wafer = detect_yolo_center(frame_global)[0]
    dx_local = wafer[0] - Target_Point[0]
    dy_local = Target_Point[1] - wafer[1]
    log(f"[DEBUG] dx = {dx_local}, dy = {dy_local}")
    if abs(dx_local) <= TOLERANCE_PX and abs(dy_local) <= TOLERANCE_PX:
        log("[Align] Within tolerance.")
        return_to_origin()
        already_moved = True
        tracking_stopped = True
        alignment_done = True
    else:
        move_stage(dx_local, dy_local)
        for _ in range(30):  # 1초 대기하며 GUI 처리
            time.sleep(0.033)
        # 정렬 다시 시도하도록 상태 초기화
        confirmed_center = None
        last_detected = None
        already_moved = False
        tracking_stopped = False
        alignment_thread = None

def main():
    global last_detected, confirm_start_time, confirmed_center
    global already_moved, tracking_stopped, show_limit_box
    global frame_global, alignment_thread, alignment_done

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)
    cv2.namedWindow("Wafer Align", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Wafer Align", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            log("[Camera] Read Fail")
            break
        frame_global = frame.copy()

        if not tracking_stopped:
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

                if confirmed_center and not already_moved and alignment_thread is None:
                    alignment_done = False
                    alignment_thread = threading.Thread(target=threaded_alignment)
                    alignment_thread.start()

        if wafer:
            cv2.circle(frame, wafer, 5, (0,255,0), -1)
        cv2.circle(frame, tuple(Target_Point), 5, (0,0,255), -1)
        if wafer:
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
            alignment_thread = None
            alignment_done = False
            log_buffer.clear()
        elif key == ord('w'):
            show_limit_box = not show_limit_box
        elif key == ord('u'):
            return_to_home_dxdy()
            log("[Manual] Returned to absolute HOME_DXDY")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
