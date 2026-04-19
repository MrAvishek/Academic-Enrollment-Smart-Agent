import os
import pickle
import numpy as np
from ml.utils.face_detection import FaceDetector
from ml.utils.embedding import FaceEmbedder

DATA_DIR = "data/raw"
SAVE_PATH = "data/embeddings/student_db.pkl"

def enroll_students():
    detector = FaceDetector()
    embedder = FaceEmbedder()
    
    student_db = {}

    print("--- Starting Enrollment Process ---")

    for student_name in os.listdir(DATA_DIR):
        student_path = os.path.join(DATA_DIR, student_name)

        if not os.path.isdir(student_path):
            continue

        print(f"Processing: {student_name}...")
        embeddings = []

        for file in os.listdir(student_path):
            img_path = os.path.join(student_path, file)
            import cv2
            img = cv2.imread(img_path)

            if img is None: continue

            face = detector.detect_face(img)
            if face is not None:
                emb = embedder.get_embedding(face)
                embeddings.append(emb)

        if embeddings:
            # We save the average (mean) embedding for the student 
            # to make comparison faster and more robust
            student_db[student_name] = np.mean(embeddings, axis=0)

    # Save the dictionary to a file
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(student_db, f)

    print(f"✅ Enrollment Complete! Database saved to {SAVE_PATH}")

if __name__ == "__main__":
    enroll_students()