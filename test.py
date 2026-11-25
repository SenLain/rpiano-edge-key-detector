import time
from math import gcd
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly, butter, sosfilt
from ai_edge_litert.interpreter import Interpreter

TARGET_SR = 16000
N_FFT = 8192
SAMPLES_TARGET = int(0.5 * TARGET_SR)  # 8000 samples @ 16kHz
CONF_THRESHOLD = 0.45
VOLUME_GATE = 0.0075                   # Blocks key-release tails and weak room reflections

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def midi_to_name(m):
    return f"{NOTE_NAMES[m % 12]}{(m // 12) - 1}"

dev_info = sd.query_devices(kind="input")
NATIVE_SR = int(dev_info["default_samplerate"])
SAMPLES_NATIVE = int(0.5 * NATIVE_SR)
HOP_NATIVE = int(0.25 * NATIVE_SR)

g = gcd(TARGET_SR, NATIVE_SR)
UP, DOWN = TARGET_SR // g, NATIVE_SR // g
print(f"Using mic: {dev_info['name']} @ {NATIVE_SR} Hz (anti-aliased {UP}/{DOWN} -> {TARGET_SR} Hz)")

# 75 Hz high-pass filter removes 50 Hz mains hum & mechanical key-hit thumps (eliminates phantom G1)
hp_sos = butter(4, 75.0, btype="highpass", fs=TARGET_SR, output="sos")

basis = np.load("log_freq_basis.npy")
window = np.hanning(SAMPLES_TARGET).astype(np.float32)

interp = Interpreter(model_path="piano_88_quant.tflite")
interp.allocate_tensors()
in_idx = interp.get_input_details()[0]["index"]
out_idx = interp.get_output_details()[0]["index"]

ring_buffer = np.zeros(SAMPLES_NATIVE, dtype=np.float32)
print("Listening... (Ctrl+C to stop)\n")

with sd.InputStream(samplerate=NATIVE_SR, channels=1, dtype="float32", blocksize=HOP_NATIVE) as stream:
    while True:
        chunk, _ = stream.read(HOP_NATIVE)
        ring_buffer = np.roll(ring_buffer, -HOP_NATIVE)
        ring_buffer[-HOP_NATIVE:] = chunk[:, 0]

        rms = float(np.sqrt(np.mean(ring_buffer ** 2)))
        bar = "#" * min(30, int(rms * 1500))

        if rms < VOLUME_GATE:
            print(f"\r[SILENT] RMS: {rms:.4f} |{bar:<30}|", end="", flush=True)
            continue

        t0 = time.perf_counter()
        wave = resample_poly(ring_buffer, UP, DOWN)[:SAMPLES_TARGET].astype(np.float32)
        wave = sosfilt(hp_sos, wave).astype(np.float32)

        peak = np.max(np.abs(wave))
        if peak > 1e-4:
            wave = wave / peak

        spec = np.abs(np.fft.rfft(wave * window, n=N_FFT)).astype(np.float32)
        feat = np.log1p(basis @ spec)
        max_f = np.max(feat)
        if max_f > 1e-6:
            feat = feat / max_f
        feat = feat.reshape(1, 176)

        interp.set_tensor(in_idx, feat)
        interp.invoke()
        probs = interp.get_tensor(out_idx)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        active_indices = np.where(probs >= CONF_THRESHOLD)[0]
        if len(active_indices) > 0:
            detected = [f"{midi_to_name(i + 21)} ({probs[i]:.2f})" for i in active_indices]
            print(f"\nRMS: {rms:.4f} | Detected: {', '.join(detected):<38} | {latency_ms:.2f} ms")
        else:
            print(f"\r[NO NOTES] RMS: {rms:.4f} |{bar:<30}|", end="", flush=True)