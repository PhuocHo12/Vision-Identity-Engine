import argparse
import time
import cv2
import os

from src.identity_engine import VisionIdentityEngine
from src.config import EngineConfig


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


def draw_identity_preview(frame, face_img, identity):
    """
    Draw matched identity image at top-left corner
    """
    if face_img is None:
        return

    h, w = 100, 100
    face_img = cv2.resize(face_img, (w, h))

    frame[40 : 40 + h, 10 : 10 + w] = face_img
    cv2.putText(
        frame,
        identity,
        (10, 40 + h + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def main():
    parser = argparse.ArgumentParser("Vision Identity Engine Demo")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--fps-limit", type=float, default=0.0, help="Limit FPS (0 = unlimited)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold")
    parser.add_argument("--save-video", type=str, default="", help="Output video path")
    parser.add_argument("--show-fps", action="store_true")

    args = parser.parse_args()

    source = parse_source(args.source)

    config = EngineConfig(recognition_threshold=args.threshold)
    engine = VisionIdentityEngine(config)
    engine.build_database("database/identities")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))

    prev_time = time.time()
    fps = 0.0

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = engine.recognize(frame)

        preview_face = None
        preview_id = ""

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            identity = r["identity"]
            score = r["score"]

            label = f"{identity} ({score:.2f})"

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

            if identity != "unknown" and preview_face is None:
                preview_face = frame[y1:y2, x1:x2].copy()
                preview_id = identity

        if preview_face is not None:
            draw_identity_preview(frame, preview_face, preview_id)

        if args.show_fps:
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            draw_fps(frame, fps)

        cv2.imshow("Vision Identity Engine", frame)

        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        if args.fps_limit > 0:
            elapsed = time.time() - start
            sleep_time = max(0, (1.0 / args.fps_limit) - elapsed)
            time.sleep(sleep_time)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()