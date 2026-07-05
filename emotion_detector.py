import sys
from typing import Any

import cv2
from fer.fer import FER


class EmotionDetector:
    """Detect facial emotions in individual OpenCV frames."""

    def __init__(self) -> None:
        self.detector = FER()

    def analyze_frame(
        self,
        frame: Any,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """
        Analyze one frame and return:
        1. A copy of the frame with annotations.
        2. Structured emotion-detection results.
        """

        annotated_frame = frame.copy()
        detections = self.detector.detect_emotions(frame)
        results: list[dict[str, Any]] = []

        for detection in detections:
            x, y, width, height = detection["box"]
            emotions = detection["emotions"]

            dominant_emotion = max(emotions, key=emotions.get)
            confidence = float(emotions[dominant_emotion])

            # FER can occasionally return negative coordinates.
            x = max(0, x)
            y = max(0, y)

            results.append(
                {
                    "emotion": dominant_emotion,
                    "confidence": confidence,
                    "box": (x, y, width, height),
                    "emotions": emotions,
                }
            )

            cv2.rectangle(
                annotated_frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            label = f"{dominant_emotion}: {confidence:.0%}"

            cv2.putText(
                annotated_frame,
                label,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return annotated_frame, results


def run_webcam() -> None:
    """Run the detector as a standalone OpenCV webcam application."""

    detector = EmotionDetector()
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open the webcam.")
        return

    print("SellSense emotion detector started.")
    print("Press Q while the camera window is selected to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Could not read a webcam frame.")
                break

            annotated_frame, _ = detector.analyze_frame(frame)

            cv2.imshow(
                "SellSense Emotion Detection",
                annotated_frame,
            )

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
    run_webcam()