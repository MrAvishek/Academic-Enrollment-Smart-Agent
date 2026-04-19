import numpy as np
import tensorflow as tf
from keras_facenet import FaceNet
import os

class FaceEmbedder:
    def __init__(self):
        """
        FaceNet model initialization.
        """
        # if not os.path.exists(model_path):
        #     raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Load the pre-trained Keras model
        self.model = FaceNet()

    def _preprocess(self, face):
        """
        Internal helper to standardize the face image.
        FaceNet requires pixels to be standardized (mean/std).
        """
        face = face.astype("float32")
        
        # Calculate mean and std for the individual face
        mean, std = face.mean(), face.std()
        
        # Prevent division by zero if the image is a solid color
        std_adj = np.maximum(std, 1e-6)
        
        face_standardized = (face - mean) / std_adj
        return face_standardized

    def get_embedding(self, face):
        """
        Converts a cropped face image into a 128 or 512-dimensional vector.
        
        Args:
            face (np.ndarray): Cropped RGB face image.
            
        Returns:
            np.ndarray: The embedding vector, or None if input is invalid.
        """
        if face is None:
            return None

        # 1. Standardize pixels
        face_prepared = self._preprocess(face)

        # 2. Add batch dimension: (160, 160, 3) -> (1, 160, 160, 3)
        face_batch = np.expand_dims(face_prepared, axis=0)

        # 3. Inference
        embedding = self.model.embeddings([face])

        # 4. Return the first (and only) result in the batch
        return embedding[0]