from __future__ import annotations

from collections import Counter
import io
from pathlib import Path
import re
import tempfile
from typing import Any
import wave

from faster_whisper import WhisperModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


FILLER_PATTERNS: dict[str, re.Pattern[str]] = {
    "um": re.compile(r"\bum+\b", re.IGNORECASE),
    "uh": re.compile(r"\buh+\b", re.IGNORECASE),
    "like": re.compile(r"\blike\b", re.IGNORECASE),
    "you know": re.compile(r"\byou\s+know\b", re.IGNORECASE),
    "i mean": re.compile(r"\bi\s+mean\b", re.IGNORECASE),
    "kind of": re.compile(r"\bkind\s+of\b", re.IGNORECASE),
    "sort of": re.compile(r"\bsort\s+of\b", re.IGNORECASE),
    "basically": re.compile(r"\bbasically\b", re.IGNORECASE),
    "actually": re.compile(r"\bactually\b", re.IGNORECASE),
}


class ToneAnalyzer:
    """
    Analyze a short speech recording.

    This version measures transcript sentiment, pace, filler words, and
    approximate pauses. It does not infer a speaker's internal emotion.
    """

    def __init__(self, model_size: str = "base.en") -> None:
        self.model_size = model_size
        self.transcriber = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def analyze_audio(self, audio_bytes: bytes) -> dict[str, Any]:
        """Transcribe WAV bytes and return communication metrics."""
        if not audio_bytes:
            raise ValueError("The audio recording is empty.")

        duration_seconds = self._get_wav_duration(audio_bytes)
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temporary_file:
                temporary_file.write(audio_bytes)
                temporary_path = Path(temporary_file.name)

            segment_generator, info = self.transcriber.transcribe(
                str(temporary_path),
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,
            )
            raw_segments = list(segment_generator)

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        segments = [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            }
            for segment in raw_segments
            if segment.text.strip()
        ]

        transcript = " ".join(segment["text"] for segment in segments).strip()
        words = re.findall(r"\b[\w']+\b", transcript)
        word_count = len(words)

        if duration_seconds <= 0 and segments:
            duration_seconds = max(segment["end"] for segment in segments)

        words_per_minute = (
            word_count / (duration_seconds / 60)
            if duration_seconds > 0
            else 0.0
        )

        filler_counts = {
            name: len(pattern.findall(transcript))
            for name, pattern in FILLER_PATTERNS.items()
        }
        filler_counts = {
            name: count
            for name, count in filler_counts.items()
            if count > 0
        }
        filler_total = sum(filler_counts.values())
        filler_rate = (
            filler_total / word_count * 100
            if word_count > 0
            else 0.0
        )

        long_pauses = self._get_long_pauses(segments)
        speech_seconds = sum(
            max(0.0, segment["end"] - segment["start"])
            for segment in segments
        )
        approximate_pause_ratio = (
            max(0.0, duration_seconds - speech_seconds) / duration_seconds
            if duration_seconds > 0
            else 0.0
        )

        sentiment_scores = self.sentiment_analyzer.polarity_scores(transcript)
        sentiment_label = self._classify_sentiment(
            sentiment_scores["compound"]
        )

        pace_label = self._classify_pace(words_per_minute)
        recommendations = self._build_recommendations(
            word_count=word_count,
            words_per_minute=words_per_minute,
            filler_rate=filler_rate,
            long_pause_count=len(long_pauses),
            sentiment_label=sentiment_label,
        )

        return {
            "transcript": transcript,
            "duration_seconds": round(duration_seconds, 1),
            "word_count": word_count,
            "words_per_minute": round(words_per_minute, 1),
            "pace_label": pace_label,
            "filler_total": filler_total,
            "filler_rate_per_100_words": round(filler_rate, 1),
            "filler_counts": filler_counts,
            "long_pause_count": len(long_pauses),
            "long_pauses": long_pauses,
            "approximate_pause_ratio": round(approximate_pause_ratio, 3),
            "sentiment_label": sentiment_label,
            "sentiment_scores": sentiment_scores,
            "language": info.language,
            "language_probability": float(info.language_probability),
            "segments": segments,
            "recommendations": recommendations,
        }

    @staticmethod
    def _get_wav_duration(audio_bytes: bytes) -> float:
        """Return WAV duration, or zero if the bytes cannot be decoded."""
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                frame_rate = wav_file.getframerate()
                if frame_rate <= 0:
                    return 0.0
                return wav_file.getnframes() / frame_rate
        except (wave.Error, EOFError):
            return 0.0

    @staticmethod
    def _get_long_pauses(
        segments: list[dict[str, Any]],
        threshold_seconds: float = 1.0,
    ) -> list[dict[str, float]]:
        pauses: list[dict[str, float]] = []

        for previous, current in zip(segments, segments[1:]):
            gap = current["start"] - previous["end"]
            if gap >= threshold_seconds:
                pauses.append(
                    {
                        "start": round(previous["end"], 2),
                        "end": round(current["start"], 2),
                        "duration": round(gap, 2),
                    }
                )

        return pauses

    @staticmethod
    def _classify_pace(words_per_minute: float) -> str:
        if words_per_minute == 0:
            return "No speech detected"
        if words_per_minute < 100:
            return "Slow"
        if words_per_minute <= 170:
            return "Balanced"
        return "Fast"

    @staticmethod
    def _classify_sentiment(compound_score: float) -> str:
        if compound_score >= 0.05:
            return "Positive"
        if compound_score <= -0.05:
            return "Negative"
        return "Neutral"

    @staticmethod
    def _build_recommendations(
        *,
        word_count: int,
        words_per_minute: float,
        filler_rate: float,
        long_pause_count: int,
        sentiment_label: str,
    ) -> list[str]:
        recommendations: list[str] = []

        if word_count == 0:
            return [
                "No clear speech was detected. Record again in a quieter "
                "environment and speak closer to the microphone."
            ]

        if words_per_minute > 170:
            recommendations.append(
                "Slow down slightly so the customer has time to process "
                "important details."
            )
        elif words_per_minute < 100:
            recommendations.append(
                "Increase the pace slightly and use shorter transitions to "
                "maintain energy."
            )
        else:
            recommendations.append(
                "Your speaking pace was within the prototype's balanced range."
            )

        if filler_rate >= 5:
            recommendations.append(
                "Replace filler words with a brief silent pause before the "
                "next point."
            )
        elif filler_rate > 0:
            recommendations.append(
                "Filler-word use was limited; keep transitions deliberate."
            )
        else:
            recommendations.append(
                "No configured filler words were detected."
            )

        if long_pause_count >= 3:
            recommendations.append(
                "Several pauses longer than one second were detected. Rehearse "
                "the transitions between your main points."
            )

        if sentiment_label == "Negative":
            recommendations.append(
                "The transcript wording leaned negative. Review the context and "
                "consider framing concerns around solutions and outcomes."
            )
        elif sentiment_label == "Positive":
            recommendations.append(
                "The transcript wording leaned positive. Keep the language "
                "specific and avoid sounding exaggerated."
            )

        return recommendations
