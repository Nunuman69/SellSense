import cv2
from fer import FER

def detect_emotion():
    detector = FER(mtcnn=True)  # Use MTCNN for better face detection
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotions
        result = detector.detect_emotions(frame)
        for face in result:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)

            # Draw rectangle and emotion text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{dominant_emotion} ({emotions[dominant_emotion]:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
