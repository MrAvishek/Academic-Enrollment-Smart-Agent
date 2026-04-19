import cv2
import numpy as np
from mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        """
        Initializes the MTCNN detector once.
        Keeping this in a class prevents reloading the model for every image.
        """
        self.detector = MTCNN()

    def detect_face(self, image, target_size=(160, 160)):
        """
        Detects, crops, and resizes a face from a BGR image.
        
        Args:
            image (np.ndarray): The input image in BGR format (OpenCV standard).
            target_size (tuple): The required output size (width, height).
            
        Returns:
            np.ndarray: The cropped and resized face in RGB, or None if no face is found.
        """
        if image is None:
            return None

        try:
            # 1. Convert BGR → RGB (MTCNN expects RGB)
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 2. Detect faces
            results = self.detector.detect_faces(rgb_img)

            if not results:
                return None

            # 3. Extract bounding box of the first face
            x, y, w, h = results[0]['box']

            # 4. Fix potential negative coordinates
            x, y = max(0, x), max(0, y)

            # 5. Crop the face
            face = rgb_img[y:y+h, x:x+w]

            # 6. Safety check for empty crops
            if face.size == 0:
                return None

            # 7. Resize to model input size (e.g., 160x160 for FaceNet)
            face_resized = cv2.resize(face, target_size)

            return face_resized

        except Exception as e:
            print(f"Error during face detection: {e}")
            return None

# --- Usage Example ---
# detector = FaceDetector()
# face = detector.detect_face(img)