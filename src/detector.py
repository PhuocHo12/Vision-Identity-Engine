from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, model_name: str, det_size=(640, 640), device="cpu"):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0 if device == "cuda" else -1,
                         det_size=det_size)

    def detect(self, frame):
        """
        Returns InsightFace Face objects
        """
        return self.app.get(frame)