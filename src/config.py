from dataclasses import dataclass
from typing import Tuple


@dataclass
class EngineConfig:
    # =========================
    # Detection
    # =========================
    det_model: str = "buffalo_s"
    det_size: Tuple[int, int] = (640, 640)

    # =========================
    # Recognition
    # =========================
    similarity_metric: str = "cosine"
    recognition_threshold: float = 0.45

    # =========================
    # Runtime
    # =========================
    device: str = "cpu"          # "cuda" if available
    fp16: bool = False           # enable half precision if supported

    # =========================
    # Performance (optional)
    # =========================
    fps_limit: float = 0.0       # 0 = unlimited
    show_fps: bool = False

    # =========================
    # Output / Debug
    # =========================
    save_video: bool = False
    log_level: str = "INFO"