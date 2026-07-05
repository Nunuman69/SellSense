from collections.abc import Callable
from datetime import datetime
import threading
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from emotion_detector import EmotionDetector
from session_tracker import SessionTracker


PROCESS_EVERY_N_FRAMES = 5
SESSION_SAMPLE_INTERVAL_SECONDS = 1.0

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
    """Load the model once and protect it for WebRTC callback access."""
    return EmotionDetector(), threading.Lock()


def get_session_tracker() -> SessionTracker:
    """Create one tracker for this browser session."""
    if "emotion_session_tracker" not in st.session_state:
        st.session_state.emotion_session_tracker = SessionTracker(
            min_sample_interval=SESSION_SAMPLE_INTERVAL_SECONDS
        )
    return st.session_state.emotion_session_tracker


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


def get_session_coaching(snapshot: dict[str, Any]) -> str:
    """Generate a cautious coaching summary from expression estimates."""
    distribution = snapshot["distribution"]

    if not distribution:
        return "No expression observations were collected."

    negative_share = sum(
        distribution.get(name, 0.0)
        for name in ("angry", "fear", "sad", "disgust")
    )

    if negative_share >= 0.30:
        return (
            "Several observations were classified as potentially negative. "
            "Review whether the seller paused, acknowledged concerns, and "
            "asked clarifying questions instead of pushing forward."
        )

    if distribution.get("neutral", 0.0) >= 0.50:
        return (
            "Most observations were neutral. Consider using more open-ended "
            "questions and checking whether the customer sees the value."
        )

    if distribution.get("happy", 0.0) >= 0.40:
        return (
            "Many observations were classified as happy. Review the moments "
            "that produced positive reactions and reinforce those benefits."
        )

    if distribution.get("surprise", 0.0) >= 0.20:
        return (
            "Surprise appeared repeatedly. Check whether important details, "
            "pricing, or expectations needed clearer explanation."
        )

    return (
        "The session contained a mixed set of expression estimates. Review "
        "the timeline alongside the conversation context before drawing "
        "conclusions."
    )


def format_duration(seconds: float) -> str:
    """Format seconds as MM:SS."""
    total_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


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
    suggestion = SHORT_SUGGESTIONS.get(
        emotion,
        "Ask a clarifying question.",
    )

    cv2.putText(
        frame,
        f"Emotion: {emotion.title()} ({confidence:.0%})",
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
    tracker: SessionTracker,
) -> Callable[[av.VideoFrame], av.VideoFrame]:
    """Create a live frame callback with reusable inference results."""
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

                primary_result = get_primary_result(state["results"])
                if primary_result is not None:
                    tracker.record(primary_result)

            except Exception:
                # Keep the stream running if one inference fails.
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


@st.fragment(run_every=1.0)
def render_session_dashboard(tracker: SessionTracker) -> None:
    """Refresh session statistics independently once per second."""
    snapshot = tracker.snapshot()

    st.subheader("Session dashboard")

    status_text = "Recording" if snapshot["active"] else "Not recording"
    if snapshot["active"]:
        st.success(f"Status: {status_text}")
    else:
        st.caption(f"Status: {status_text}")

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric(
        "Duration",
        format_duration(snapshot["duration_seconds"]),
    )
    metric_two.metric(
        "Samples",
        snapshot["total_samples"],
    )
    metric_three.metric(
        "Dominant estimate",
        (
            snapshot["dominant_emotion"].title()
            if snapshot["dominant_emotion"]
            else "—"
        ),
    )

    events = snapshot["events"]
    if not events:
        st.info(
            "Start a session and keep a face visible to collect observations."
        )
        return

    distribution_frame = pd.DataFrame(
        {
            "Expression": [
                emotion.title()
                for emotion in snapshot["distribution"].keys()
            ],
            "Percentage": [
                share * 100
                for share in snapshot["distribution"].values()
            ],
        }
    ).set_index("Expression")

    st.markdown("#### Expression distribution")
    st.bar_chart(distribution_frame)

    events_frame = pd.DataFrame(events)
    confidence_frame = events_frame[
        ["elapsed_seconds", "confidence"]
    ].copy()
    confidence_frame["confidence"] *= 100
    confidence_frame = confidence_frame.set_index("elapsed_seconds")

    st.markdown("#### Detection confidence over time")
    st.line_chart(confidence_frame)

    recent_events = events_frame.tail(10).copy()
    recent_events["emotion"] = recent_events["emotion"].str.title()
    recent_events["confidence"] = recent_events["confidence"].map(
        lambda value: f"{value:.1%}"
    )
    recent_events = recent_events.rename(
        columns={
            "elapsed_seconds": "Seconds",
            "emotion": "Expression",
            "confidence": "Confidence",
        }
    )

    st.markdown("#### Recent observations")
    st.dataframe(
        recent_events[["Seconds", "Expression", "Confidence"]],
        hide_index=True,
        use_container_width=True,
    )

    if not snapshot["active"]:
        st.markdown("#### Session summary")
        st.write(
            f"Average model confidence: "
            f"**{snapshot['average_confidence']:.1%}**"
        )
        st.info(get_session_coaching(snapshot))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download session observations (CSV)",
            data=tracker.to_csv(),
            file_name=f"sellsense_session_{timestamp}.csv",
            mime="text/csv",
            on_click="ignore",
            use_container_width=True,
        )


detector, detector_lock = load_model_resources()
tracker = get_session_tracker()

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
        "Start the video, then begin a tracking session. SellSense stores "
        "only timestamps, expression labels, and confidence values in memory."
    )

    control_one, control_two, control_three = st.columns(3)

    if control_one.button(
        "Start session",
        type="primary",
        use_container_width=True,
    ):
        tracker.start()

    if control_two.button(
        "End session",
        use_container_width=True,
        disabled=not tracker.snapshot()["active"],
    ):
        tracker.stop()

    if control_three.button(
        "Reset",
        use_container_width=True,
    ):
        tracker.reset()

    video_callback = build_video_callback(
        detector,pp
        detector_lock,
        tracker,
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
        "SellSense analyzes every fifth frame and records at most one "
        "observation per second during an active session."
    )

    render_session_dashboard(tracker)

else:
    st.subheader("Snapshot analysis")

    camera_image = st.camera_input("Capture a customer expression")

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
            st.error("SellSense could not process the captured image.")

        else:
            with st.spinner("Analyzing facial expression..."):
                with detector_lock:
                    annotated_frame, results = detector.analyze_frame(frame)

            st.image(
                annotated_frame,
                channels="BGR",
                caption="Analyzed image",
                use_container_width=True,
            )

            primary_result = get_primary_result(results)

            if primary_result is None:
                st.warning(
                    "No face was detected. Try better lighting and keep "
                    "your face clearly visible."
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
                            f"**{emotion_name.title()}**: {score:.1%}"
                        )
