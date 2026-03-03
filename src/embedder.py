import numpy as np

class FaceEmbedder:
    def extract(self, faces):
        """
        faces: list of InsightFace Face objects
        return: list of np.ndarray embeddings
        """
        embeddings = []
        for face in faces:
            embeddings.append(face.embedding.astype("float32"))
        return embeddings