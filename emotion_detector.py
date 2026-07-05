import sys

import cv2
from fer.fer import FER


def detect_emotion() -> None:
    """Open the webcam and display real-time facial emotion predictions."""

    # Start with FER's default face detector.
    # We can enable MTCNN after the basic version works.
    detector = FER()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open the webcam.")
        print("Check Windows camera permissions and close other camera apps.")
        return

    print("SellSense emotion detector started.")
    print("Press Q while the camera window is selected to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Could not read a frame from the webcam.")
                break

            detections = detector.detect_emotions(frame)

            for detection in detections:
                x, y, width, height = detection["box"]
                emotions = detection["emotions"]

                dominant_emotion = max(emotions, key=emotions.get)
                confidence = emotions[dominant_emotion]

                # Prevent negative coordinates from drawing outside the frame.
                x = max(0, x)
                y = max(0, y)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2,
                )

                label = f"{dominant_emotion}: {confidence:.0%}"

                cv2.putText(
                    frame,
                    label,
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("SellSense Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nEmotion detection stopped.")

    except Exception as error:
        print(f"Unexpected error: {error}")
        sys.exit(1)

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_emotion()