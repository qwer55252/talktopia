"""Build a transcript and a seek timeline from already-saved audio."""

from __future__ import annotations

import json
import math
import re
import wave
from pathlib import Path

from .files import ResultError, artifact_path


AUDIBLE_ACTIONS = {"speak", "backchanneling", "hesitation", "correction", "interruption"}


def wav_duration(path: Path) -> tuple[float, int]:
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate(), audio.getframerate()
    except (OSError, EOFError, wave.Error, ZeroDivisionError) as error:
        raise ResultError(f"{path.name}: WAV 길이를 확인할 수 없습니다.") from error


def read_records(path: Path, warnings: list[str], label: str = "발화 기록") -> list[dict]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        warnings.append(f"{label}이 없습니다. 저장된 대화 텍스트를 표시합니다.")
        return rows
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError()
            rows.append(row)
        except ValueError:
            # A producer may still be writing the last line. Keep readable rows.
            warnings.append(f"{label} {index}행을 읽을 수 없습니다.")
            for row in rows:
                row["_incomplete_log"] = True
            return rows
    return rows


def actions(source: dict) -> list[dict]:
    result = []
    for turn, messages in enumerate(source.get("messages") or [], start=1):
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, (list, tuple)) or len(message) != 3:
                continue
            sender, receiver, text = message
            if sender == "Environment" or receiver != "Environment" or not isinstance(text, str):
                continue
            text = re.sub(r"^\[private to \[.*?\]\]\s*", "", text.strip())
            if not text or text == "did nothing":
                continue
            spoken = text.startswith('said: "')
            action_type = "speak" if spoken else "action"
            if text.startswith('backchanneled: "'):
                action_type = "backchanneling"
            bracket = re.match(r"^\[([a-z -]+)\]", text)
            if bracket:
                action_type = bracket[1]
            result.append({
                "turn": turn,
                "speaker": sender,
                "text": text[7:-1] if spoken and text.endswith('"') else text,
                "action": action_type,
                "start": None,
                "end": None,
                "committed": True,
            })
    return result


def transcript(root: Path, row: dict, source: dict) -> dict:
    warnings = []
    try:
        speech = read_records(artifact_path(root, row.get("speech"), ".jsonl"), warnings)
    except ResultError as error:
        warnings.append(str(error))
        speech = []
    audio_available = False
    duration = None
    rate = 24000
    if row.get("conversation_audio"):
        try:
            path = artifact_path(root, row["conversation_audio"], ".wav")
            duration, rate = wav_duration(path)
            audio_available = True
        except ResultError as error:
            warnings.append(str(error))

    is_duplex = (
        row.get("interaction_mode") == "surface5-full-duplex"
        or "CascadedDuplexAgent" in (source.get("agent_classes") or [])
        or any("utterance_id" in item for item in speech)
    )
    if is_duplex:
        entries = []
        for item in speech:
            # Received speech can exist before a semantic commit; label it explicitly.
            text = item.get("asr_text") or item.get("received_text") or ""
            start, end = item.get("start_ms"), item.get("end_ms")
            valid = (
                audio_available
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end))
                and 0 <= start <= end <= duration * 1000 + 1
                and not item.get("_incomplete_log")
            )
            entries.append({
                "speaker": item.get("speaker", ""),
                "text": text or "(수신 텍스트 없음)",
                "action": item.get("action_type", "speak"),
                "start": start / 1000 if valid else None,
                "end": end / 1000 if valid else None,
                "committed": bool(item.get("commit_id")),
            })
        if not speech:
            # Original messages retain committed received speech even if the
            # optional timing file is unavailable.
            entries.extend(item for item in actions(source) if item["action"] in AUDIBLE_ACTIONS)
        # Nonverbal commits have no speech record. Their event timestamp places
        # them between the audible utterances, instead of in commit order.
        if row.get("events"):
            try:
                events = read_records(artifact_path(root, row["events"], ".jsonl"), warnings, "이벤트 기록")
            except ResultError as error:
                warnings.append(str(error))
                events = []
            for event in events:
                if event.get("event_type") != "action_committed":
                    continue
                for speaker, action in event.get("actions", {}).items():
                    if action.get("action_type") in AUDIBLE_ACTIONS | {"none"}:
                        continue
                    timestamp = event.get("timestamp_ms")
                    valid = (
                        audio_available and isinstance(timestamp, (int, float))
                        and math.isfinite(timestamp) and 0 <= timestamp <= duration * 1000
                    )
                    entries.append({
                        "speaker": speaker, "text": action.get("argument", ""),
                        "action": action.get("action_type", "action"),
                        "start": timestamp / 1000 if valid else None,
                        "end": timestamp / 1000 if valid else None,
                        "committed": True,
                    })
            if not events:
                entries.extend(item for item in actions(source) if item["action"] not in AUDIBLE_ACTIONS)
        else:
            entries.extend(item for item in actions(source) if item["action"] not in AUDIBLE_ACTIONS)
        entries.sort(key=lambda item: (item["start"] is None, item["start"] or 0))
        if not entries:
            entries = actions(source)
    else:
        timed = {}
        cursor = 0.0
        audible_count = 0
        valid = audio_available and not any(item.get("_incomplete_log") for item in speech)
        for item in speech:
            if item.get("status") != "completed" or not item.get("wav_path"):
                continue
            try:
                length, sample_rate = wav_duration(
                    artifact_path(root, item["wav_path"], ".wav")
                )
                if sample_rate != rate:
                    raise ResultError("발화와 전체 음성의 sample rate가 다릅니다.")
            except ResultError as error:
                warnings.append(str(error))
                valid = False
                continue
            if audible_count:
                cursor += round(rate * 0.3) / rate
            timed[(item.get("turn"), item.get("speaker"))] = {
                "text": item.get("asr_text"),
                "start": cursor,
                "end": cursor + length,
            }
            audible_count += 1
            cursor += length
        if not audible_count or duration is None or abs(cursor - duration) > 1 / rate:
            valid = False
        entries = actions(source)
        if not entries:
            entries = [
                {
                    "turn": item.get("turn"), "speaker": item.get("speaker", ""),
                    "text": item.get("asr_text") or "(수신 텍스트 없음)",
                    "action": "speak", "start": None, "end": None, "committed": False,
                }
                for item in speech if item.get("status") == "completed"
            ]
        for entry in entries:
            timing = timed.get((entry.get("turn"), entry["speaker"]))
            if timing and entry["action"] == "speak":
                if timing["text"] is not None:
                    entry["text"] = timing["text"] or "(수신 텍스트 없음)"
                if valid:
                    entry.update(start=timing["start"], end=timing["end"])
        if audio_available and not valid:
            warnings.append("발화 길이와 전체 음성의 시간을 확인할 수 없어 대사 클릭 이동을 해제했습니다.")
    return {
        "entries": entries,
        "duration": duration,
        "audio_available": audio_available,
        "mode": "surface5-full-duplex" if is_duplex else "round-robin",
        "warnings": warnings,
    }
