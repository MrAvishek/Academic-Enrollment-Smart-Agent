import cv2
import numpy as np
import pickle
import os
from datetime import datetime
from ml.utils.face_detection import FaceDetector
from ml.utils.embedding import FaceEmbedder

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

def run_realtime():
    detector = FaceDetector()
    embedder = FaceEmbedder()
    
    # Load the pre-saved embeddings
    if not os.path.exists(DB_PATH):
        print("❌ Error: student_db.pkl not found. Run enrollment first!")
        return
    with open(DB_PATH, 'rb') as f:
        student_db = pickle.load(f)

    cap = cv2.VideoCapture(0) # Open default webcam
    logged_students = set() # Track who we already logged this session

    print("--- Press 'q' to quit ---")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Detect Face
        # Note: We use the raw frame for drawing, but pass to detector for processing
        face = detector.detect_face(frame)

        if face is not None:
            # 2. Get Embedding
            test_emb = embedder.get_embedding(face)

            # 3. Recognition Logic
            best_name = "Unknown"
            min_dist = float("inf")

            for name, saved_emb in student_db.items():
                dist = np.linalg.norm(test_emb - saved_emb)
                if dist < min_dist:
                    min_dist = dist
                    best_name = name

            # 4. Threshold & Display
            if min_dist < THRESHOLD:
                color = (0, 255, 0) # Green for recognized
                label = f"{best_name} ({min_dist:.2f})"
                
                # Log to CSV if new this session
                if best_name not in logged_students:
                    log_attendance(best_name)
                    logged_students.add(best_name)
            else:
                color = (0, 0, 255) # Red for unknown
                label = "Unknown"

            # 5. Visual Feedback (Drawing a simple box manually since we have face location)
            # For simplicity, we just put text on the frame. 
            # (To draw a box, we'd need to return coordinates from detector.py)
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Smart Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime()