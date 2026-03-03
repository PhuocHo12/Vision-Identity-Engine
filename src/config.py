from dataclasses import dataclass

@dataclass
class EngineConfig:
    det_model: str = "buffalo_s"
    det_size: tuple = (640, 640)

    similarity_metric: str = "cosine"
    recognition_threshold: float = 0.45

    device: str = "cpu"   # "cuda" if available