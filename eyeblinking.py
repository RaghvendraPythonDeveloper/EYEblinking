import cv2
import mediapipe as mp
import time
import psutil
import os
import threading
import numpy as np

def get_system_uptime():
    """ Returns system uptime in seconds. """
    return time.time() - psutil.boot_time()

def save_uptime_data():
    """ Logs uptime to console and appends it to a file with a daily cumulative total. """
    while True:
        uptime_seconds = get_system_uptime()
        uptime_hours = int(uptime_seconds // 3600)
        uptime_minutes = int((uptime_seconds % 3600) // 60)
        uptime_seconds = int(uptime_seconds % 60)

        today = time.strftime('%Y-%m-%d')
        log_entry = f"{today} - {uptime_hours} hr : {uptime_minutes} min : {uptime_seconds} sec\n"
        print(f"Screen Time: {log_entry}", end="\r", flush=True)

        log_file = "uptime_log.txt"
        if os.path.exists(log_file):
            with open(log_file, "r") as file:
                lines = file.readlines()
            
            for i, line in enumerate(lines):
                if line.startswith(today):
                    lines[i] = log_entry
                    break
            else:
                lines.append(log_entry)

            with open(log_file, "w") as file:
                file.writelines(lines)
        else:
            with open(log_file, "w") as file:
                file.write(log_entry)
        
        time.sleep(60)  # Update every minute

def eye_aspect_ratio(eye_landmarks):
    """Calculate Eye Aspect Ratio (EAR) to detect blinks."""
    left_dist = ((eye_landmarks[1][0] - eye_landmarks[5][0])**2 + (eye_landmarks[1][1] - eye_landmarks[5][1])**2) ** 0.5
    right_dist = ((eye_landmarks[2][0] - eye_landmarks[4][0])**2 + (eye_landmarks[2][1] - eye_landmarks[4][1])**2) ** 0.5
    horizontal_dist = ((eye_landmarks[0][0] - eye_landmarks[3][0])**2 + (eye_landmarks[0][1] - eye_landmarks[3][1])**2) ** 0.5
    return (left_dist + right_dist) / (2.0 * horizontal_dist)

def save_blink_data(blink_count):
    """Save blink data to a text file."""
    with open("blink_data.txt", "a") as file:
        file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Blinks: {blink_count}\n")
        print(f"The no. of blinks you have done are {blink_count}")

def calculate_face_distance(face_landmarks):
    """Estimate distance based on the width between eyes."""
    left_eye = face_landmarks.landmark[33]  # Left eye corner
    right_eye = face_landmarks.landmark[263]  # Right eye corner
    return ((right_eye.x - left_eye.x) ** 2 + (right_eye.y - left_eye.y) ** 2) ** 0.5

def blink_detection():
    """Detect eye blinks and monitor distance from screen."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    start_time = time.time()
    blink_count = 0
    eye_closed_frames = 0
    last_blink_time = time.time()
    blink_warning_shown = False
    distance_warning_shown = False
    TOO_CLOSE_THRESHOLD = 0.20

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ Auto-Detect Low Light & Boost Contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 60:  # If too dark, enhance contrast
            frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=40)

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [33, 160, 158, 133, 153, 144]]
                right_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in [362, 385, 387, 263, 373, 380]]
                avg_ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                # ✅ Lower EAR threshold for glasses users
                if avg_ear < 0.28:
                    eye_closed_frames += 1
                else:
                    if eye_closed_frames >= 3:
                        blink_count += 1
                        last_blink_time = time.time()
                        blink_warning_shown = False
                    eye_closed_frames = 0

                # ✅ Monitor distance from screen
                eye_distance = calculate_face_distance(face_landmarks)
                if eye_distance > TOO_CLOSE_THRESHOLD and not distance_warning_shown:
                    print("\033[91m🚨 Warning: Too close to screen!\033[0m")
                    distance_warning_shown = True
                else:
                    distance_warning_shown = False

        elapsed_time = time.time() - start_time
        if elapsed_time > 60:
            save_blink_data(blink_count)
            blink_count = 0
            start_time = time.time()

        # ✅ Blink reminder after 6 seconds instead of 5
        if time.time() - last_blink_time > 6 and not blink_warning_shown:
            print("🚨 You haven't blinked for 6 seconds! Blink now!")
            blink_warning_shown = True

        cv2.putText(frame, f"Blinks: {blink_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Eye Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    threading.Thread(target=save_uptime_data, daemon=True).start()
    blink_detection()
