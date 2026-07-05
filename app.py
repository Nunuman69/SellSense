from collections.abc import Callable
import threading
from typing import Any

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from emotion_detector import EmotionDetector


PROCESS_EVERY_N_FRAMES = 5


FULL_SUGGESTIONS = {
    "happy": (
        "The customer appears receptive. Reinforce the product's main "
        "benefit and move toward the next step."
    ),
    "neutral": (
        "The customer appears neutral. Ask an open-ended question to "
        "better understand their interest."
    ),
    "sad": (
        "Slow down and acknowledge the customer's concerns before "
        "continuing the presentation."
    ),
    "angry": (
        "Avoid pushing the sale. Listen carefully, acknowledge the "
        "concern, and clarify the issue."
    ),
    "fear": (
        "The customer may appear uncertain. Explain the product, price, "
        "or process more clearly and provide reassurance."
    ),
    "surprise": (
        "Pause and check whether the information was expected or "
        "requires additional explanation."
    ),
    "disgust": (
        "The customer may be reacting negatively. Ask which aspect does "
        "not meet their expectations."
    ),
}


SHORT_SUGGESTIONS = {
    "happy": "Reinforce the benefit and discuss the next step.",
    "neutral": "Ask an open-ended question.",
    "sad": "Slow down and acknowledge their concern.",
    "angry": "Listen first and avoid pushing the sale.",
    "fear": "Clarify the process and provide reassurance.",
    "surprise": "Pause and offer further explanation.",
    "disgust": "Ask what does not meet their expectations.",
}


st.set_page_config(
    page_title="SellSense",
    page_icon="📊",
    layout="centered",
)


@st.cache_resource
def load_model_resources() -> tuple[EmotionDetector, Any]:
    """
    Load the emotion model once and protect it with a lock because
    WebRTC frame processing occurs outside Streamlit's main thread.
    """

    return EmotionDetector(), threading.Lock()


def get_primary_result(
    results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the result belonging to the largest detected face."""

    if not results:
        return None

    return max(
        results,
        key=lambda result: result["box"][2] * result["box"][3],
    )


def get_sales_suggestion(emotion: str) -> str:
    """Return the full recommendation for an emotion."""

    return FULL_SUGGESTIONS.get(
        emotion,
        "Continue listening and ask a clarifying question.",
    )


def draw_live_summary(
    frame: Any,
    results: list[dict[str, Any]],
) -> Any:
    """Draw the current primary emotion and suggestion on the video."""

    primary_result = get_primary_result(results)

    panel_width = min(frame.shape[1] - 20, 620)

    cv2.rectangle(
        frame,
        (10, 10),
        (10 + panel_width, 95),
        (0, 0, 0),
        -1,
    )

    if primary_result is None:
        cv2.putText(
            frame,
            "No face detected",
            (25, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Face the camera and improve the lighting.",
            (25, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        return frame

    emotion = primary_result["emotion"]
    confidence = primary_result["confidence"]

    emotion_text = f"Emotion: {emotion.title()} ({confidence:.0%})"

    suggestion = SHORT_SUGGESTIONS.get(
        emotion,
        "Ask a clarifying question.",
    )

    cv2.putText(
        frame,
        emotion_text,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        suggestion,
        (25, 77),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    return frame


def build_video_callback(
    detector: EmotionDetector,
    detector_lock: Any,
) -> Callable[[av.VideoFrame], av.VideoFrame]:
    """
    Create a WebRTC callback with its own frame counter and latest result.

    Emotion detection is intentionally not performed on every frame
    because model inference is considerably slower than webcam capture.
    """

    state: dict[str, Any] = {
        "frame_count": 0,
        "results": [],
    }

    def video_frame_callback(
        frame: av.VideoFrame,
    ) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        state["frame_count"] += 1

        should_analyze = (
            state["frame_count"] == 1
            or state["frame_count"] % PROCESS_EVERY_N_FRAMES == 0
        )

        if should_analyze:
            try:
                with detector_lock:
                    state["results"] = detector.detect(image)

            except Exception:
                # Keep the stream alive if one inference fails.
                state["results"] = []

        annotated_frame = detector.annotate_frame(
            image,
            state["results"],
        )

        annotated_frame = draw_live_summary(
            annotated_frame,
            state["results"],
        )

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24",
        )

    return video_frame_callback


detector, detector_lock = load_model_resources()

st.title("SellSense")
st.caption("Real-time AI-assisted communication analysis")

st.warning(
    "Facial-expression scores are probabilistic estimates. They should "
    "not be treated as proof of what someone is actually feeling."
)

mode = st.radio(
    "Analysis mode",
    ("Live video", "Snapshot"),
    horizontal=True,
)


if mode == "Live video":
    st.subheader("Live emotion analysis")

    st.write(
        "Press **START**, allow camera access, and face the camera. "
        "SellSense will display expression estimates and communication "
        "suggestions directly on the video."
    )

    video_callback = build_video_callback(
        detector,
        detector_lock,
    )

    webrtc_streamer(
        key="sellsense-live-video",
        video_frame_callback=video_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
            },
            "audio": False,
        },
        media_toggle_controls=False,
    )

    st.caption(
        "For better performance, SellSense analyzes every fifth frame "
        "and reuses the latest result between analyses."
    )


else:
    st.subheader("Snapshot analysis")

    camera_image = st.camera_input(
        "Capture a customer expression"
    )

    if camera_image is not None:
        image_bytes = np.asarray(
            bytearray(camera_image.getvalue()),
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            st.error(
                "SellSense could not process the captured image."
            )

        else:
            with st.spinner("Analyzing facial expression..."):
                with detector_lock:
                    annotated_frame, results = (
                        detector.analyze_frame(frame)
                    )

            st.image(
                annotated_frame,
                channels="BGR",
                caption="Analyzed image",
                use_container_width=True,
            )

            primary_result = get_primary_result(results)

            if primary_result is None:
                st.warning(
                    "No face was detected. Try better lighting and "
                    "keep your face clearly visible."
                )

            else:
                emotion = primary_result["emotion"]
                confidence = primary_result["confidence"]

                first_column, second_column = st.columns(2)

                first_column.metric(
                    "Detected emotion",
                    emotion.title(),
                )

                second_column.metric(
                    "Confidence",
                    f"{confidence:.0%}",
                )

                st.subheader("Suggested response")
                st.info(get_sales_suggestion(emotion))

                with st.expander("View all emotion scores"):
                    sorted_emotions = sorted(
                        primary_result["emotions"].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )

                    for emotion_name, score in sorted_emotions:
                        st.write(
                            f"**{emotion_name.title()}**: "
                            f"{score:.1%}"
                        )