import argparse
import cv2

from src.identity_engine import VisionIdentityEngine
from src.config import EngineConfig


def parse_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def main():
    parser = argparse.ArgumentParser(
        description="Vision Identity Engine - Real-time Face Recognition"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: camera index (0), video file path, or RTSP URL",
    )

    args = parser.parse_args()
    source = parse_source(args.source)

    # Initialize engine
    config = EngineConfig()
    engine = VisionIdentityEngine(config)

    # Build identity database
    engine.build_database("data/identities")

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = engine.recognize(frame)

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            label = f"{r['identity']} ({r['score']:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Vision Identity Engine", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()