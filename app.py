import cv2
import numpy as np
import streamlit as st

from emotion_detector import EmotionDetector


st.set_page_config(
    page_title="SellSense",
    page_icon="📊",
    layout="centered",
)


@st.cache_resource
def load_emotion_detector() -> EmotionDetector:
    """Load the FER model once instead of reloading on every interaction."""
    return EmotionDetector()


def get_sales_suggestion(emotion: str) -> str:
    """Return a basic sales recommendation for the detected emotion."""

    suggestions = {
        "happy": (
            "The customer appears receptive. Reinforce the product's "
            "main benefit and move toward the next step."
        ),
        "neutral": (
            "The customer appears neutral. Ask an open-ended question "
            "to better understand their interest."
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
            "The customer may feel uncertain. Explain the product, cost, "
            "or process more clearly and provide reassurance."
        ),
        "surprise": (
            "Pause and check whether the information was expected or "
            "requires further explanation."
        ),
        "disgust": (
            "The customer may be reacting negatively. Ask what aspect "
            "does not meet their expectations."
        ),
    }

    return suggestions.get(
        emotion,
        "Continue listening and ask a clarifying question.",
    )


st.title("SellSense")
st.caption("AI-assisted emotion analysis for virtual sales communication")

st.write(
    "Take a picture using your webcam. SellSense will analyze the "
    "visible facial expression and provide a basic communication suggestion."
)

camera_image = st.camera_input("Capture a customer expression")

if camera_image is not None:
    image_bytes = np.asarray(
        bytearray(camera_image.getvalue()),
        dtype=np.uint8,
    )

    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        st.error("SellSense could not process the captured image.")

    else:
        detector = load_emotion_detector()

        with st.spinner("Analyzing facial expression..."):
            annotated_frame, results = detector.analyze_frame(frame)

        st.subheader("Analysis")

        st.image(
            annotated_frame,
            channels="BGR",
            caption="Analyzed image",
            use_container_width=True,
        )

        if not results:
            st.warning(
                "No face was detected. Try again with better lighting "
                "and keep your face clearly visible."
            )

        else:
            # Use the largest detected face as the primary subject.
            primary_result = max(
                results,
                key=lambda result: (
                    result["box"][2] * result["box"][3]
                ),
            )

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