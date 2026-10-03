"""Run the EXP-001 acceptance criteria and record what was actually measured.

This harness executes criteria 1-6 and 8 on **synthetic fixtures** because no real
Tamil recording exists yet. Criterion 7 (plotting) and the README's requirement
that the criteria be measured on a real Tamil recording are therefore recorded as
*pending*, never as passed. It writes ``results.json`` next to itself.

Run from anywhere:

    python experiments/001_audio_pipeline/verify_criteria.py

Fixtures are written under ``artifacts/exp001/fixtures/`` (gitignored); no audio
enters version control.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tamil_voice.audio.features import DEFAULT_N_MELS, StftConfig, log_mel_spectrogram  # noqa: E402
from tamil_voice.audio.io import (  # noqa: E402
    AudioLoadError,
    AudioValidationError,
    ISSUE_CLIPPED,
    ISSUE_EMPTY,
    ISSUE_INFINITE,
    ISSUE_LOW_AMPLITUDE,
    ISSUE_NAN,
    ISSUE_SILENT,
    load_audio,
)
from tamil_voice.audio.resampling import resample_audio, resample_to_canonical  # noqa: E402
from tamil_voice.common.config import load_yaml  # noqa: E402
from tamil_voice.vad.detector import VadConfig, detect_speech  # noqa: E402
from tamil_voice.vad.postprocess import SegmentConfig, build_segments  # noqa: E402

FIXTURES = ROOT / "artifacts" / "exp001" / "fixtures"
CANONICAL_SR = 16_000

CONFIG = load_yaml(Path(__file__).parent / "config.yaml")
TONE_HZ = float(CONFIG["fixture"]["tone_hz"])
TONE_AMP = float(CONFIG["fixture"]["amplitude"])
FIXTURE_DURATION = float(CONFIG["fixture"]["duration_seconds"])


def sine(seconds: float, freq: float, sr: int, amp: float = TONE_AMP) -> np.ndarray:
    t = np.arange(int(round(seconds * sr)), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_frequency(waveform: np.ndarray, sample_rate: int) -> float:
    data = np.asarray(waveform, dtype=np.float64)
    windowed = data * np.hanning(data.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(data.size, 1.0 / sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


def write_wav(name: str, waveform: np.ndarray, sample_rate: int, subtype: str = "FLOAT") -> Path:
    path = FIXTURES / name
    sf.write(str(path), waveform, sample_rate, subtype=subtype)
    return path


def speech_like(seconds: float, sr: int, freq: float = 180.0) -> np.ndarray:
    """A voiced vowel approximation: harmonics shaped by a slow envelope."""
    t = np.arange(int(round(seconds * sr)), dtype=np.float64) / sr
    envelope = 0.5 * (1.0 - np.cos(2 * np.pi * 4.0 * t))
    signal = np.zeros_like(t)
    for harmonic in range(1, 7):
        signal += np.sin(2 * np.pi * freq * harmonic * t) / harmonic
    return (0.35 * envelope * signal).astype(np.float32)


# --------------------------------------------------------------------- criteria


def criterion_1() -> dict[str, Any]:
    cfg = CONFIG["criterion_1"]
    write_wav("c1_48k_stereo.wav", np.stack([sine(FIXTURE_DURATION, TONE_HZ, 48_000)] * 2, axis=1), 48_000)
    write_wav("c1_22k_mono.wav", sine(FIXTURE_DURATION, TONE_HZ, 22_050), 22_050)

    a = resample_to_canonical(load_audio(FIXTURES / "c1_48k_stereo.wav"))
    b = resample_to_canonical(load_audio(FIXTURES / "c1_22k_mono.wav"))
    n = min(a.num_frames, b.num_frames)
    peak = max(float(np.max(np.abs(a.waveform))), 1e-9) if n else 1.0
    difference = float(np.max(np.abs(a.waveform[:n] - b.waveform[:n]))) if n else float("inf")
    freq_a = dominant_frequency(a.waveform, a.sample_rate)
    freq_b = dominant_frequency(b.waveform, b.sample_rate)
    measured = {
        "sr48k_result": {"sample_rate": a.sample_rate, "channels": a.channels, "frames": a.num_frames},
        "sr22k_result": {"sample_rate": b.sample_rate, "channels": b.channels, "frames": b.num_frames},
        "max_waveform_difference": difference,
        "relative_waveform_difference": difference / peak,
        "dominant_hz": [freq_a, freq_b],
    }
    passed = a.sample_rate == b.sample_rate == CANONICAL_SR
    passed = passed and a.channels == 1 and b.channels == 1
    passed = passed and abs(a.num_frames - b.num_frames) <= cfg["max_frame_difference"]
    passed = passed and (difference / peak) <= cfg["max_relative_waveform_difference"]
    passed = passed and abs(freq_a - freq_b) <= cfg["max_frequency_difference_hz"]
    tolerance = (
        f"same 16 kHz mono; frame diff <={cfg['max_frame_difference']}; "
        f"relative diff <={cfg['max_relative_waveform_difference']}; "
        f"dominant freq within {cfg['max_frequency_difference_hz']} Hz"
    )
    return {"status": "pass" if passed else "fail", "tolerance": tolerance, "measured": measured}


def criterion_2() -> dict[str, Any]:
    rates = [8_000, 22_050, 44_100, 48_000]
    results = {}
    for rate in rates:
        path = write_wav(f"c2_{rate}.wav", sine(1.0, TONE_HZ, rate), rate)
        out = resample_to_canonical(load_audio(path))
        results[str(rate)] = {"sample_rate": out.sample_rate, "frames": out.num_frames}

    canonical = load_audio(write_wav("c2_16000.wav", sine(1.0, TONE_HZ, CANONICAL_SR), CANONICAL_SR))
    passthrough = resample_audio(canonical)
    results["passthrough_is_same_object"] = passthrough is canonical

    passed = all(v["sample_rate"] == CANONICAL_SR for k, v in results.items() if k.isdigit())
    passed = passed and bool(results["passthrough_is_same_object"])
    return {"status": "pass" if passed else "fail", "tolerance": "exact 16 kHz; identity on passthrough", "measured": results}


def criterion_3() -> dict[str, Any]:
    tolerance_hz = float(CONFIG["criterion_3"]["max_frequency_error_hz"])
    results = {}
    for rate in [8_000, 22_050, 44_100, 48_000]:
        path = write_wav(f"c3_{rate}.wav", sine(2.0, TONE_HZ, rate), rate)
        out = resample_to_canonical(load_audio(path))
        measured_hz = dominant_frequency(out.waveform, out.sample_rate)
        results[str(rate)] = {"dominant_hz": measured_hz, "error_hz": abs(measured_hz - TONE_HZ)}
    passed = all(v["error_hz"] <= tolerance_hz for v in results.values())
    return {"status": "pass" if passed else "fail", "tolerance": f"<= {tolerance_hz} Hz", "measured": results}


def _codes_for(path: Path) -> dict[str, Any]:
    try:
        audio = load_audio(path)
    except AudioValidationError as exc:
        return {"raised": "AudioValidationError", "codes": sorted(i.code for i in exc.report.issues)}
    except AudioLoadError:
        return {"raised": "AudioLoadError", "codes": []}
    return {"raised": None, "codes": sorted(i.code for i in audio.report.issues) if audio.report else []}


def criterion_4() -> dict[str, Any]:
    corrupt = FIXTURES / "c4_corrupt.wav"
    corrupt.write_bytes(b"RIFF\x00\x00this is not audio data")
    empty = FIXTURES / "c4_empty.wav"
    empty.write_bytes(b"")

    nan_signal = sine(0.5, TONE_HZ, CANONICAL_SR).copy()
    nan_signal[100] = np.nan
    write_wav("c4_nan.wav", nan_signal, CANONICAL_SR)
    inf_signal = sine(0.5, TONE_HZ, CANONICAL_SR).copy()
    inf_signal[100] = np.inf
    write_wav("c4_inf.wav", inf_signal, CANONICAL_SR)
    write_wav("c4_clipped.wav", sine(0.5, TONE_HZ, CANONICAL_SR, amp=1.1), CANONICAL_SR)
    write_wav("c4_near_silent.wav", sine(0.5, TONE_HZ, CANONICAL_SR, amp=1e-6), CANONICAL_SR)
    write_wav("c4_silent.wav", np.zeros(CANONICAL_SR // 2, dtype=np.float32), CANONICAL_SR)

    found = {
        "corrupt": _codes_for(corrupt),
        "empty_file": _codes_for(empty),
        "nan": _codes_for(FIXTURES / "c4_nan.wav"),
        "infinite": _codes_for(FIXTURES / "c4_inf.wav"),
        "clipped": _codes_for(FIXTURES / "c4_clipped.wav"),
        "near_silent": _codes_for(FIXTURES / "c4_near_silent.wav"),
        "silent": _codes_for(FIXTURES / "c4_silent.wav"),
    }

    def has(name: str, code: str) -> bool:
        return code in found[name]["codes"]

    passed = (
        found["corrupt"]["raised"] == "AudioLoadError"
        and found["empty_file"]["raised"] == "AudioLoadError"
        and has("nan", ISSUE_NAN)
        and has("infinite", ISSUE_INFINITE)
        and has("clipped", ISSUE_CLIPPED)
        and has("near_silent", ISSUE_LOW_AMPLITUDE)
        and has("silent", ISSUE_SILENT)
    )
    return {"status": "pass" if passed else "fail", "tolerance": "expected issue code present", "measured": found}


def criterion_5() -> dict[str, Any]:
    signal = sine(FIXTURE_DURATION, TONE_HZ, CANONICAL_SR)
    config = StftConfig()
    mel = log_mel_spectrogram(signal, config)
    frames, bins = mel.shape
    expected_frames = 1 + signal.size // config.hop_length
    measured = {
        "shape": [int(frames), int(bins)],
        "n_mels": int(bins),
        "expected_n_frames": int(expected_frames),
        "win_length": config.win_length,
        "hop_length": config.hop_length,
    }
    passed = bins == DEFAULT_N_MELS == 80 and frames == expected_frames
    return {"status": "pass" if passed else "fail", "tolerance": "n_frames == 1 + N//hop", "measured": measured}


def criterion_6() -> dict[str, Any]:
    cfg = CONFIG["criterion_6"]
    sr = CANONICAL_SR
    layout = [(0.5, "silence"), (0.4, "speech"), (0.3, "silence"), (0.5, "speech"),
              (0.4, "silence"), (0.3, "speech"), (0.4, "silence")]
    pieces = []
    truth = []
    cursor = 0.0
    for seconds, kind in layout:
        if kind == "speech":
            pieces.append(speech_like(seconds, sr))
            truth.append((cursor, cursor + seconds))
        else:
            pieces.append(np.zeros(int(round(seconds * sr)), dtype=np.float32))
        cursor += seconds
    waveform = np.concatenate(pieces)
    total_duration = float(waveform.size / sr)

    vad_config = VadConfig()
    segment_config = SegmentConfig()
    result = detect_speech(waveform, vad_config)
    segments = build_segments(result, segment_config, total_duration=total_duration)

    speech_mask = np.zeros(waveform.size, dtype=bool)
    for start, end in truth:
        speech_mask[int(start * sr):int(end * sr)] = True
    detected_mask = np.zeros(waveform.size, dtype=bool)
    for segment in segments:
        detected_mask[int(segment.start * sr):int(segment.end * sr)] = True

    # The pipeline deliberately extends segments by pad + hangover. Allow that
    # margin when judging false positives; anything detected further from real
    # speech than the configured margin is a genuine error.
    margin = segment_config.pad_seconds + vad_config.hangover_frames * vad_config.hop_seconds
    allowed_mask = np.zeros(waveform.size, dtype=bool)
    for start, end in truth:
        allowed_mask[max(0, int((start - margin) * sr)):int((end + margin) * sr)] = True

    true_samples = int(speech_mask.sum())
    overlap = int(np.logical_and(speech_mask, detected_mask).sum())
    detected_samples = int(detected_mask.sum())
    false_samples = int(np.logical_and(detected_mask, ~speech_mask).sum())
    beyond_samples = int(np.logical_and(detected_mask, ~allowed_mask).sum())
    coverage = overlap / true_samples if true_samples else 0.0
    leakage = false_samples / detected_samples if detected_samples else 1.0
    leakage_beyond_margin = beyond_samples / detected_samples if detected_samples else 1.0

    measured = {
        "num_segments": len(segments),
        "segments": [{"start": round(s.start, 4), "end": round(s.end, 4)} for s in segments],
        "ground_truth": [{"start": round(a, 4), "end": round(b, 4)} for a, b in truth],
        "coverage": round(coverage, 4),
        "leakage_raw": round(leakage, 4),
        "edge_margin_seconds": round(margin, 4),
        "leakage_beyond_margin": round(leakage_beyond_margin, 4),
    }
    passed = coverage >= cfg["coverage_min"] and leakage_beyond_margin <= cfg["leakage_beyond_margin_max"]
    tolerance = (
        f"coverage >={cfg['coverage_min']}; beyond configured pad+hangover margin, "
        f"<={cfg['leakage_beyond_margin_max']} false positive"
    )
    return {"status": "pass" if passed else "fail", "tolerance": tolerance, "measured": measured}


def criterion_8() -> dict[str, Any]:
    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)

    pytest = run([sys.executable, "-m", "pytest"])
    ruff = run([sys.executable, "-m", "ruff", "check", "src", "tests"])
    mypy = run([sys.executable, "-m", "mypy"])

    match = re.search(r"(\d+) passed", pytest.stdout)
    measured = {
        "pytest_exit": pytest.returncode,
        "pytest_passed": int(match.group(1)) if match else None,
        "ruff_exit": ruff.returncode,
        "ruff_tail": ruff.stdout.strip().splitlines()[-1] if ruff.stdout.strip() else "",
        "mypy_exit": mypy.returncode,
        "mypy_tail": mypy.stdout.strip().splitlines()[-1] if mypy.stdout.strip() else "",
    }
    passed = pytest.returncode == 0 and ruff.returncode == 0 and mypy.returncode == 0
    return {"status": "pass" if passed else "fail", "tolerance": "all three gates exit 0", "measured": measured}


CRITERIA = [
    (1, "48 kHz stereo and 22.05 kHz mono converge on 16 kHz mono", criterion_1),
    (2, "8/22.05/44.1/48 kHz resample; 16 kHz passes through untouched", criterion_2),
    (3, "resampling preserves the dominant frequency", criterion_3),
    (4, "corrupt/empty/NaN/Inf/clipped/near-silent are reported", criterion_4),
    (5, "mel features are (n_frames, 80) with 25 ms/10 ms geometry", criterion_5),
    (6, "VAD segments cover speech and nothing else, to tolerance", criterion_6),
    (7, "waveform/spectrogram/mel/VAD regions are plotted", None),
    (8, "unit tests exist and pytest/ruff/mypy all pass", criterion_8),
]


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    results = []
    for number, description, fn in CRITERIA:
        if fn is None:
            results.append({
                "id": number,
                "description": description,
                "status": "pending",
                "tolerance": None,
                "measured": None,
                "reason": "requires one real Tamil recording in data/raw/speech/, which does not exist yet",
            })
            print(f"criterion {number}: PENDING (no real Tamil recording)")
            continue
        outcome = fn()
        outcome.update({"id": number, "description": description})
        results.append(outcome)
        print(f"criterion {number}: {outcome['status'].upper()}")

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    pending = sum(1 for r in results if r["status"] == "pending")

    payload = {
        "experiment": "EXP-001",
        "title": "Audio Pipeline Foundation",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "input": "synthetic fixtures only",
        "real_recording_required": True,
        "criteria": results,
        "summary": {"passed": passed, "failed": failed, "pending": pending},
        "honesty_note": (
            "Criterion 7 is pending and the README specifies the criteria be measured on a "
            "real Tamil recording. Criteria 1-6 and 8 are measured on synthetic fixtures only "
            "and must be re-run on the real recording before EXP-001 is accepted. No value here "
            "is fabricated; every number was produced by this run."
        ),
    }
    (Path(__file__).parent / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n{passed} pass, {failed} fail, {pending} pending -> results.json written")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
