import cv2
from src.identity_engine import VisionIdentityEngine
from src.config import EngineConfig

config = EngineConfig()
engine = VisionIdentityEngine(config)

engine.build_database("data/identities")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = engine.recognize(frame)

    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        label = f"{r['identity']} ({r['score']})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Vision Identity Engine", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()