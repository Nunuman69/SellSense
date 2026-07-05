import sys
from typing import Any

import cv2
from fer.fer import FER


class EmotionDetector:
    """Detect and annotate facial emotions in OpenCV frames."""

    def __init__(self) -> None:
        self.detector = FER()

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        """Detect faces and return structured emotion results."""

        detections = self.detector.detect_emotions(frame)
        results: list[dict[str, Any]] = []

        for detection in detections:
            raw_x, raw_y, raw_width, raw_height = detection["box"]

            x = max(0, int(raw_x))
            y = max(0, int(raw_y))
            width = max(0, int(raw_width))
            height = max(0, int(raw_height))

            emotions = {
                name: float(score)
                for name, score in detection["emotions"].items()
            }

            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]

            results.append(
                {
                    "emotion": dominant_emotion,
                    "confidence": confidence,
                    "box": (x, y, width, height),
                    "emotions": emotions,
                }
            )

        return results

    def annotate_frame(
        self,
        frame: Any,
        results: list[dict[str, Any]],
    ) -> Any:
        """Draw face boxes and emotion labels on a frame."""

        annotated_frame = frame.copy()

        for result in results:
            x, y, width, height = result["box"]
            emotion = result["emotion"]
            confidence = result["confidence"]

            cv2.rectangle(
                annotated_frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            label = f"{emotion.title()}: {confidence:.0%}"

            cv2.putText(
                annotated_frame,
                label,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return annotated_frame

    def analyze_frame(
        self,
        frame: Any,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Detect emotions and return an annotated frame and results."""

        results = self.detect(frame)
        annotated_frame = self.annotate_frame(frame, results)

        return annotated_frame, results


def run_webcam() -> None:
    """Run SellSense as a standalone OpenCV webcam application."""

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