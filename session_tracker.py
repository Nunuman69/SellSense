from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import io
import threading
import time
from typing import Any


class SessionTracker:
    """Thread-safe in-memory tracker for live emotion observations."""

    def __init__(self, min_sample_interval: float = 1.0) -> None:
        if min_sample_interval <= 0:
            raise ValueError("min_sample_interval must be greater than zero.")

        self.min_sample_interval = min_sample_interval
        self._lock = threading.Lock()
        self._active = False
        self._started_at_monotonic: float | None = None
        self._started_at_utc: datetime | None = None
        self._ended_at_utc: datetime | None = None
        self._last_sample_at_monotonic: float | None = None
        self._events: list[dict[str, Any]] = []

    def start(self) -> None:
        """Start a fresh tracking session."""
        now_monotonic = time.monotonic()
        now_utc = datetime.now(timezone.utc)

        with self._lock:
            self._active = True
            self._started_at_monotonic = now_monotonic
            self._started_at_utc = now_utc
            self._ended_at_utc = None
            self._last_sample_at_monotonic = None
            self._events = []

    def stop(self) -> None:
        """Stop tracking while preserving collected observations."""
        with self._lock:
            if self._active:
                self._ended_at_utc = datetime.now(timezone.utc)
            self._active = False

    def reset(self) -> None:
        """Clear the current session and all collected observations."""
        with self._lock:
            self._active = False
            self._started_at_monotonic = None
            self._started_at_utc = None
            self._ended_at_utc = None
            self._last_sample_at_monotonic = None
            self._events = []

    def record(self, result: dict[str, Any]) -> None:
        """Record one primary-face emotion result at a controlled rate."""
        now_monotonic = time.monotonic()

        with self._lock:
            if not self._active or self._started_at_monotonic is None:
                return

            if (
                self._last_sample_at_monotonic is not None
                and now_monotonic - self._last_sample_at_monotonic
                < self.min_sample_interval
            ):
                return

            emotion = str(result.get("emotion", "unknown"))
            confidence = float(result.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            self._events.append(
                {
                    "elapsed_seconds": round(
                        now_monotonic - self._started_at_monotonic,
                        2,
                    ),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "emotion": emotion,
                    "confidence": confidence,
                }
            )
            self._last_sample_at_monotonic = now_monotonic

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of the session and calculated statistics."""
        with self._lock:
            events = [event.copy() for event in self._events]
            active = self._active
            started_at_monotonic = self._started_at_monotonic
            started_at_utc = self._started_at_utc
            ended_at_utc = self._ended_at_utc

        if started_at_monotonic is None:
            duration_seconds = 0.0
        elif active:
            duration_seconds = time.monotonic() - started_at_monotonic
        elif events:
            duration_seconds = float(events[-1]["elapsed_seconds"])
        else:
            duration_seconds = 0.0

        counts = Counter(event["emotion"] for event in events)
        total_samples = len(events)

        distribution = (
            {
                emotion: count / total_samples
                for emotion, count in counts.items()
            }
            if total_samples
            else {}
        )

        dominant_emotion = counts.most_common(1)[0][0] if counts else None

        average_confidence = (
            sum(event["confidence"] for event in events) / total_samples
            if total_samples
            else 0.0
        )

        return {
            "active": active,
            "started_at_utc": (
                started_at_utc.isoformat() if started_at_utc else None
            ),
            "ended_at_utc": (
                ended_at_utc.isoformat() if ended_at_utc else None
            ),
            "duration_seconds": round(duration_seconds, 1),
            "total_samples": total_samples,
            "dominant_emotion": dominant_emotion,
            "average_confidence": average_confidence,
            "distribution": distribution,
            "events": events,
        }

    def to_csv(self) -> str:
        """Return collected observations as CSV text."""
        events = self.snapshot()["events"]
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "elapsed_seconds",
                "timestamp_utc",
                "emotion",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(events)
        return output.getvalue()
