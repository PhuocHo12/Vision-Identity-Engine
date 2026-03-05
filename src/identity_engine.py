from .detector import FaceDetector
from .embedder import FaceEmbedder
from .matcher import IdentityMatcher
from .database import IdentityDatabase
from .config import EngineConfig


class VisionIdentityEngine:
    def __init__(self, config: EngineConfig):
        self.config = config

        self.detector = FaceDetector(
            model_name=config.det_model,
            det_size=config.det_size,
            device=config.device
        )
        self.embedder = FaceEmbedder()
        self.matcher = IdentityMatcher(config.recognition_threshold)
        self.database = IdentityDatabase()

    def build_database(self, folder):
        self.database.load_from_folder(
            folder,
            self.detector,
            self.embedder
        )

    def recognize(self, frame):
        results = []
        faces = self.detector.detect(frame)
        embeddings = self.embedder.extract(faces)

        for face, emb in zip(faces, embeddings):
            name, score = self.matcher.match(
                emb,
                self.database.embeddings
            )

            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "identity": name,
                "score": round(score, 3),
                "reference_face": self.database.reference_face.get(name, None)
            })

        return results