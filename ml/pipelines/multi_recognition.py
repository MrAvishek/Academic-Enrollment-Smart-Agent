import os
import cv2
import numpy as np

from ml.utils.face_detection import FaceDetector
from ml.utils.embedding import FaceEmbedder

detector = FaceDetector()
embedder = FaceEmbedder()



DATA_DIR = "data/raw"

# ---------------------------
# Load all students
# ---------------------------

def load_all_students(data_dir):
    student_db = {}

    for student_name in os.listdir(data_dir):
        student_path = os.path.join(data_dir, student_name)

        # Skip files like test.jpg
        if not os.path.isdir(student_path):
            continue

        embeddings = []

        for file in os.listdir(student_path):
            img_path = os.path.join(student_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            face = detector.detect_face(img)
            if face is None:
                continue

            emb = embedder.get_embedding(face)
            embeddings.append(emb)

        if embeddings:
            student_db[student_name] = embeddings

    return student_db

# ---------------------------
# Find best match
# ---------------------------

def recognize(test_embedding, db_path="data/embeddings/student_db.pkl"):
    # 1. Load the pre-saved database
    with open(db_path, 'rb') as f:
        student_db = pickle.load(f)

    best_student = "Unknown"
    min_dist = 0.7  # Threshold

    for name, saved_embedding in student_db.items():
        dist = np.linalg.norm(test_embedding - saved_embedding)
        if dist < min_dist:
            min_dist = dist
            best_student = name

    return best_student, min_dist

# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    print("Loading student database...")
    student_db = load_all_students(DATA_DIR)

    for k, v in student_db.items():
        print(f"{k}: {len(v)} embeddings")

    # Load test image
    test_img = cv2.imread("data/raw/test.jpg")
    
    if test_img is None:
        print("Could not find test.jpg")
        exit()

    face = detector.detect_face(test_img)

    if face is None:
        print("No face detected in test image")
        exit()

    test_embedding = embedder.get_embedding(face)

    student, distance = recognize(test_embedding, student_db)

    print(f"\nPredicted: {student}")
    print(f"Distance: {distance:.4f}")

    # Threshold (Tuned for Euclidean distance)
    if distance < 0.7:
        print("✅ Recognized")
    else:
        print("❌ Unknown person")