import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from math import gcd

import numpy as np
import requests
import sounddevice as sd
from ai_edge_litert.interpreter import Interpreter
from azure.iot.device import Message
from azure.iot.device.aio import IoTHubDeviceClient
from scipy.signal import butter, resample_poly, sosfilt
from questdb import Sender, TimestampNanos

import config

# --- AUDIO & DSP CONSTANTS ---
TARGET_SR = 16000
N_FFT = 8192
SAMPLES_TARGET = int(0.5 * TARGET_SR)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# --- DEFAULT DEVICE TWIN SETTINGS ---
DEFAULT_IS_LISTENING = False
DEFAULT_CONF_THRESHOLD = 0.45
DEFAULT_VOLUME_GATE = 0.0075
DEFAULT_PIANO_KEYS = 88

client = None
is_listening = DEFAULT_IS_LISTENING
conf_threshold = DEFAULT_CONF_THRESHOLD
volume_gate = DEFAULT_VOLUME_GATE
piano_keys = DEFAULT_PIANO_KEYS

is_recording = False
note_counts = np.zeros(12, dtype=int)
first_pc = None
last_pc = None
last_result = None

# ------------ IOT HUB ------------

async def connect(connection_string):
    client = IoTHubDeviceClient.create_from_connection_string(connection_string)
    await client.connect()
    return client

async def disconnect():
    global client

    if client:
        await client.disconnect()
        client = None

def get_desired_properties(twin):
    return twin.get("desired", {})

def set_piano_keys(raw_value):
    global piano_keys

    try:
        candidate = int(raw_value)
    except (TypeError, ValueError):
        return

    piano_keys = min(config.STANDARD_PIANO_SIZES, key=lambda size: abs(size - candidate))

def load_twin_config(twin):
    global conf_threshold, volume_gate

    desired = get_desired_properties(twin)

    conf_threshold = float(desired.get("confThreshold", DEFAULT_CONF_THRESHOLD))
    volume_gate = float(desired.get("volumeGate", DEFAULT_VOLUME_GATE))
    set_piano_keys(desired.get("pianoKeys", DEFAULT_PIANO_KEYS))

async def twin_patch_handler(patch):
    global conf_threshold, volume_gate

    if "confThreshold" in patch:
        conf_threshold = float(patch["confThreshold"])
    if "volumeGate" in patch:
        volume_gate = float(patch["volumeGate"])
    if "pianoKeys" in patch:
        set_piano_keys(patch["pianoKeys"])

async def send_session_to_iothub(data):
    if not client:
        return
    message = Message(json.dumps(data))
    message.content_encoding = "utf-8"
    message.content_type = "application/json"
    message.message_id = str(uuid.uuid4())
    await client.send_message(message)


# -------- QUESTDB --------
def save_to_questdb(payload, notes_played):
    try:
        with Sender.from_conf(config.QUESTDB_CONF) as sender:
            sender.row(
                "piano_sessions",
                symbols={
                    "session_id": payload["session_id"],
                    "device_id": payload["device_id"],
                },
                columns={
                    "total_notes": int(payload["total_notes"]),
                    "notes_played": notes_played,
                    "synced": client is not None,
                },
                at=TimestampNanos.now(),
            )
            sender.flush()
    except Exception:
        pass


# ---- Audio & TFLite ----

def midi_to_name(midi_num):
    return f"{NOTE_NAMES[midi_num % 12]}{(midi_num // 12) - 1}"

def init_audio_pipeline():
    dev_info = sd.query_devices(kind="input")
    native_sr = int(dev_info["default_samplerate"])
    samples_native = int(0.5 * native_sr)
    hop_native = int(0.25 * native_sr)

    g = gcd(TARGET_SR, native_sr)
    up, down = TARGET_SR // g, native_sr // g

    hp_sos = butter(4, 75.0, btype="highpass", fs=TARGET_SR, output="sos")
    basis = np.load(config.BASIS_PATH)
    window = np.hanning(SAMPLES_TARGET).astype(np.float32)

    interpreter = Interpreter(model_path=str(config.MODEL_PATH))
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    return {
        "mic_info": f"{dev_info['name']} ({native_sr} Hz -> {TARGET_SR} Hz)",
        "native_sr": native_sr,
        "samples_native": samples_native,
        "hop_native": hop_native,
        "up": up,
        "down": down,
        "hp_sos": hp_sos,
        "basis": basis,
        "window": window,
        "interpreter": interpreter,
        "in_idx": in_idx,
        "out_idx": out_idx,
    }

def extract_features(ring_buffer, pipeline):
    wave = resample_poly(ring_buffer, pipeline["up"], pipeline["down"])[:SAMPLES_TARGET].astype(np.float32)
    wave = sosfilt(pipeline["hp_sos"], wave).astype(np.float32)

    peak = np.max(np.abs(wave))
    if peak > 1e-4:
        wave = wave / peak

    spec = np.abs(np.fft.rfft(wave * pipeline["window"], n=N_FFT)).astype(np.float32)
    feat = np.log1p(pipeline["basis"] @ spec)
    max_f = np.max(feat)
    if max_f > 1e-6:
        feat = feat / max_f
    return feat.reshape(1, 176)

def predict_notes(features, pipeline):
    interp = pipeline["interpreter"]
    interp.set_tensor(pipeline["in_idx"], features)
    interp.invoke()
    return interp.get_tensor(pipeline["out_idx"])[0]

def format_active_notes(probs):
    active_indices = np.where(probs >= conf_threshold)[0]
    detected = []
    valid_midis = []

    for idx in active_indices:
        midi_num = int(idx + 21)
        valid_midis.append(midi_num)
        detected.append({"note": midi_to_name(midi_num), "conf": round(float(probs[idx]), 2)})

    return detected, valid_midis


# --- SESSION RECORDING & CLOUD PAYLOAD ---

def start_recording():
    global is_recording, note_counts, first_pc, last_pc
    is_recording = True
    note_counts = np.zeros(12, dtype=int)
    first_pc = None
    last_pc = None

def accumulate_session_notes(midi_numbers):
    global first_pc, last_pc
    for midi_num in midi_numbers:
        pc = midi_num % 12
        note_counts[pc] += 1
        if first_pc is None:
            first_pc = pc
        last_pc = pc

def format_notes_summary():
    return ", ".join(
        f"{NOTE_NAMES[i]}:{int(note_counts[i])}"
        for i in range(12)
        if note_counts[i] > 0
    )

def build_session_payload():
    total = int(np.sum(note_counts))
    pcts = np.round(note_counts / float(total), 4)
    payload = {
        "session_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_id": config.DEVICE_ID,
        "total_notes": total,
    }
    for i, name in enumerate(NOTE_NAMES):
        payload[f"pct_{name}"] = float(pcts[i])
    payload["first_pc"] = first_pc
    payload["last_pc"] = last_pc
    return payload

def preview_cloud_key_mock(payload):
    """Temporary local preview until Azure Function + AutoML endpoint is deployed."""
    scores = [
        ("8A - A Minor", round(min(0.95, 0.35 + (payload["pct_A"] + payload["pct_C"] + payload["pct_E"]) * 0.7), 2)),
        ("8B - C Major", round(min(0.95, 0.32 + (payload["pct_C"] + payload["pct_E"] + payload["pct_G"]) * 0.7), 2)),
        ("9B - G Major", round(min(0.95, 0.20 + (payload["pct_G"] + payload["pct_B"] + payload["pct_D"]) * 0.6), 2)),
    ]
    scores.sort(key=lambda item: item[1], reverse=True)
    return [{"key": k, "confidence": c} for k, c in scores]

async def stop_recording():
    global is_recording, last_result
    if not is_recording:
        return

    is_recording = False
    if int(np.sum(note_counts)) == 0:
        return

    notes_played = format_notes_summary()
    payload = build_session_payload()

    await asyncio.to_thread(save_to_questdb, payload, notes_played)
    await send_session_to_iothub(payload)

    last_result = {
        "session_id": payload["session_id"],
        "total_notes": payload["total_notes"],
        "notes_played": notes_played,
        "synced": client is not None,
        "mock_cloud_predictions": preview_cloud_key_mock(payload),
    }


# --- IPC FILE BRIDGE ---

def write_json_atomic(path, data):
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        for attempt in range(10):
            try:
                os.replace(tmp_path, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.03)
    finally:
        tmp_path.unlink(missing_ok=True)

async def handle_ui_command():
    global is_listening
    if not config.COMMAND_FILE.exists():
        return
    try:
        with open(config.COMMAND_FILE, "r", encoding="utf-8") as f:
            cmd = json.load(f)
        config.COMMAND_FILE.unlink(missing_ok=True)

        action = cmd.get("action")
        if action == "start":
            start_recording()
        elif action == "stop":
            await stop_recording()
        elif action == "set_listening":
            is_listening = bool(cmd.get("value", False))
        elif action == "set_piano_keys":
            set_piano_keys(cmd.get("value"))
    except Exception:
        pass

def publish_live_state(mic_info, rms, latency_ms, detected):
    write_json_atomic(
        config.STATE_FILE,
        {
            "mic_info": mic_info,
            "rms": round(rms, 4),
            "latency_ms": round(latency_ms, 2),
            "detected": detected,
            "recording": is_recording,
            "recorded_notes_count": int(np.sum(note_counts)),
            "is_listening": is_listening,
            "iot_connected": client is not None,
            "config": {
                "confThreshold": conf_threshold,
                "volumeGate": volume_gate,
                "pianoKeys": piano_keys,
            },
            "last_result": last_result,
        },
    )


# ----------- MAIN ------------

async def main():
    global client
    
    try:
        client = await connect(config.IOT_CONNECTION_STRING)
        if client:
            twin = await client.get_twin()
            load_twin_config(twin)
            client.on_twin_desired_properties_patch_received = twin_patch_handler

        pipeline = init_audio_pipeline()
        ring_buffer = np.zeros(pipeline["samples_native"], dtype=np.float32)
        hop = pipeline["hop_native"]

        with sd.InputStream(samplerate=pipeline["native_sr"], channels=1, dtype="float32", blocksize=hop) as stream:
            while True:
                await handle_ui_command()

                chunk, _ = await asyncio.to_thread(stream.read, hop)
                ring_buffer = np.roll(ring_buffer, -hop)
                ring_buffer[-hop:] = chunk[:, 0]

                rms = float(np.sqrt(np.mean(ring_buffer ** 2)))
                detected = []
                latency_ms = 0.0

                if is_listening and rms >= volume_gate:
                    t0 = time.perf_counter()
                    features = extract_features(ring_buffer, pipeline)
                    probs = predict_notes(features, pipeline)
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    detected, midis = format_active_notes(probs)
                    if is_recording and midis:
                        accumulate_session_notes(midis)

                publish_live_state(pipeline["mic_info"], rms, latency_ms, detected)
    finally:
        await disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass