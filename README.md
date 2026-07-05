# SellSense

SellSense is a real-time AI-assisted communication prototype designed to support virtual sales conversations. It analyzes visible facial expressions from webcam video and provides simple, rule-based communication suggestions to the seller.

> **Important:** Facial-expression scores are probabilistic model estimates. They should not be treated as proof of a person's emotions, intentions, or mental state.

## Current Status

SellSense currently supports:

- Continuous browser-based webcam analysis using Streamlit WebRTC
- Snapshot-based facial-expression analysis
- Standalone OpenCV webcam mode
- Face detection and FER emotion scoring
- Dominant-emotion and confidence display
- Basic rule-based sales suggestions
- Reusable frame-by-frame emotion-detection logic

Tone, speech, and sentiment analysis are planned but are not implemented yet.

## Features

### Live Video Analysis

The live mode streams webcam video through the browser and periodically runs emotion detection. The latest result is displayed directly on the video with:

- Face bounding boxes
- Dominant emotion
- Confidence score
- A short communication suggestion

### Snapshot Analysis

The snapshot mode allows the user to capture one image and view:

- The annotated image
- The dominant detected emotion
- Confidence percentage
- A suggested response
- A breakdown of all emotion scores

### Standalone Webcam Mode

The emotion detector can also run independently in a local OpenCV window without starting Streamlit.

## Technology Stack

- Python 3.11
- Streamlit
- streamlit-webrtc
- FER
- TensorFlow
- OpenCV
- PyAV
- NumPy

## Project Structure

```text
SellSense/
├── app.py                 # Streamlit interface and WebRTC integration
├── emotion_detector.py    # Emotion detection and frame annotation
├── tone_analyzer.py       # Placeholder for future audio analysis
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Files excluded from Git
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Nunuman69/SellSense.git
cd SellSense
```

### 2. Create a virtual environment

On Windows:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 3. Install the dependencies

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Start the Streamlit application

```cmd
python -m streamlit run app.py
```

Open the local URL shown in the terminal, allow camera access, and select either **Live video** or **Snapshot** mode.

## Standalone Emotion Detector

To run the OpenCV webcam detector directly:

```cmd
python emotion_detector.py
```

Press **Q** while the webcam window is selected to stop the detector.

## How It Works

```text
Webcam input
    ↓
Streamlit WebRTC or snapshot capture
    ↓
OpenCV frame conversion
    ↓
FER facial-expression analysis
    ↓
Dominant emotion and confidence score
    ↓
Rule-based communication suggestion
```

In live mode, model inference is performed on selected frames rather than every captured frame to improve responsiveness. The latest result is reused between analyses.

## Detected Expression Categories

The current FER model may return scores for:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Privacy and Responsible Use

The current prototype does not intentionally save webcam frames, captured snapshots, or emotion results to persistent storage.

SellSense should be used only with informed consent. Its predictions can be affected by lighting, camera position, facial movement, model bias, image quality, and other environmental factors.

The application is intended as a communication-assistance experiment—not as a system for hiring, surveillance, medical assessment, or high-stakes decision-making.

## Current Limitations

- Facial expressions do not always represent a person's actual emotions
- Results may fluctuate between frames
- Performance depends on lighting and camera quality
- Live analysis can be CPU-intensive
- The largest detected face is treated as the primary subject
- Suggestions are rule-based rather than context-aware
- Audio, transcription, tone, and sentiment analysis are not implemented
- Session history and reports are not implemented

## Roadmap

- [x] Standalone webcam emotion detection
- [x] Snapshot analysis in Streamlit
- [x] Live WebRTC emotion analysis
- [x] Basic sales-response suggestions
- [ ] Track emotion changes during a session
- [ ] Generate an end-of-session summary
- [ ] Add microphone capture and transcription
- [ ] Add voice tone and sentiment analysis
- [ ] Combine video and audio signals
- [ ] Add automated tests and performance benchmarks
- [ ] Improve deployment and WebRTC configuration
- [ ] Add user-controlled privacy and data-retention settings

## Development Note

The project currently uses Python 3.11 and a pinned OpenCV package to reduce dependency conflicts. Avoid installing multiple OpenCV distributions in the same virtual environment because they share the same `cv2` namespace.
