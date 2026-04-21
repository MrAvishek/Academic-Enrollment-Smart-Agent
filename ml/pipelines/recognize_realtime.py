import cv2
import numpy as np
import pickle
import os
import threading
import queue
from datetime import datetime
from ml.utils.face_detection import FaceDetector
from ml.utils.embedding import FaceEmbedder
from backend.app.db.attendance_repo import AttendanceRepository

# Configuration
DB_PATH = "data/embeddings/student_db.pkl"
ATTENDANCE_FILE = "logs/attendance.csv"
THRESHOLD = 0.7

def log_attendance(name):
    """Writes the name and timestamp to a CSV file if not already logged today."""
    os.makedirs("logs", exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    
    # Check if file exists, if not, write header
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w") as f:
            f.write("Name,Date,Time\n")

    # Simple logic: Only log if the person hasn't been seen in the current session
    # (In a real app, you'd check the file to see if they marked in for the day)
    with open(ATTENDANCE_FILE, "a") as f:
        f.write(f"{name},{date_str},{time_str}\n")
    print(f"📝 Logged: {name} at {time_str}")

# def run_realtime():
#     detector = FaceDetector()
#     embedder = FaceEmbedder()
    
#     # Load the pre-saved embeddings
#     if not os.path.exists(DB_PATH):
#         print("❌ Error: student_db.pkl not found. Run enrollment first!")
#         return
#     with open(DB_PATH, 'rb') as f:
#         student_db = pickle.load(f)

#     cap = cv2.VideoCapture(0) # Open default webcam
#     cap.set(cv2.CAP_PROP_FPS, 10) 
#     logged_students = set() # Track who we already logged this session

#     print("--- Press 'q' to quit ---")

#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         # Reduce Frame
#         frame = cv2.resize(frame, (320,240))
#         # 1. Detect Face
#         # Note: We use the raw frame for drawing, but pass to detector for processing
#         face = detector.detect_face(frame)

#         if face is not None:
#             # 2. Get Embedding
#             test_emb = embedder.get_embedding(face)

#             # 3. Recognition Logic
#             best_name = "Unknown"
#             min_dist = float("inf")

#             for name, saved_emb in student_db.items():
#                 dist = np.linalg.norm(test_emb - saved_emb)
#                 if dist < min_dist:
#                     min_dist = dist
#                     best_name = name

#             # 4. Threshold & Display
#             if min_dist < THRESHOLD:
#                 color = (0, 255, 0) # Green for recognized
#                 label = f"{best_name} ({min_dist:.2f})"
                
#                 # Log to CSV if new this session
#                 if best_name not in logged_students:
#                     log_attendance(best_name)
#                     logged_students.add(best_name)
#             else:
#                 color = (0, 0, 255) # Red for unknown
#                 label = "Unknown"

#             # 5. Visual Feedback (Drawing a simple box manually since we have face location)
#             # For simplicity, we just put text on the frame. 
#             # (To draw a box, we'd need to return coordinates from detector.py)
#             cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#         cv2.imshow("Smart Attendance System", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

def capture_frames(cap, frame_queue ):
    while True:
        ret, frame= cap.read()
        if not ret:
            continue
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

def process_frames(frame_queue, result_queue, detector, embedder, student_db, db_repo):
    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        frame = cv2.resize(frame, (320, 240))

        face = detector.detect_face(frame)

        label = "Unknown"
        color = (0, 0, 255)  # Red default

        if face is not None:
            test_emb = embedder.get_embedding(face)

            best_name = "Unknown"
            min_dist = float("inf")

            for name, saved_emb in student_db.items():
                dist = np.linalg.norm(test_emb - saved_emb)
                if dist < min_dist:
                    min_dist = dist
                    best_name = name

            if min_dist < THRESHOLD:
                label = f"{best_name} ({min_dist:.2f})"
                color = (0, 255, 0)  # Green

                # --- AGENTIC DATABASE LOGGING ---
                try:
                    # Convert distance to a confidence percentage
                    confidence = float(1 - min_dist)
                    # The repo handles the 30-minute duplicate check internally!
                    db_repo.log_presence(best_name, confidence)
                except Exception as e:
                    print(f"⚠️ Database Log Failed: {e}")
            else:
                label = "Unknown"
                color = (0, 0, 255) 

        if not result_queue.full():
            result_queue.put((frame, label, color))


def run_realtime():
    detector = FaceDetector()
    embedder = FaceEmbedder()

    # 1. Initialize the Database Repository Agent
    db_repo = AttendanceRepository()

    if not os.path.exists(DB_PATH):
        print("❌ student_db.pkl not found")
        return

    with open(DB_PATH, 'rb') as f:
        student_db = pickle.load(f)

    cap = cv2.VideoCapture(0)

    frame_queue = queue.Queue(maxsize=5)
    result_queue = queue.Queue(maxsize=5)

    logged_students = set()

    # Threads
    t1 = threading.Thread(target=capture_frames, args=(cap, frame_queue), daemon=True)
    t2 = threading.Thread(target=process_frames, args=(
            frame_queue, result_queue, detector, embedder, student_db, db_repo
        ), daemon=True)

    t1.start()
    t2.start()

    print("--- Press 'q' to quit ---")

    while True:
        if not result_queue.empty():
            frame, label, color = result_queue.get()

            cv2.putText(frame, label, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Smart Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()
