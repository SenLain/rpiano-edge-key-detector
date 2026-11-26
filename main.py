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
from azure.iot.device import Message, MethodResponse
from azure.iot.device.aio import IoTHubDeviceClient
from scipy.signal import butter, resample_poly, sosfilt

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
# Default settings
is_listening = DEFAULT_IS_LISTENING
conf_threshold = DEFAULT_CONF_THRESHOLD
volume_gate = DEFAULT_VOLUME_GATE
piano_keys = DEFAULT_PIANO_KEYS


# ------------ IOT HUB ------------

async def connect(connection_string):
    global client

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
        canditate = int(raw_value)
    except (TypeError, ValueError):
        return

    piano_keys = min(config.STANDARD_PIANO_SIZES, key=lambda size: abs(size - canditate))

def load_twin_config(twin):
    global conf_threshold, volume_gate, piano_keys    

    desired = get_desired_properties(twin)

    conf_threshold = float(desired.get("confThreshold", DEFAULT_CONF_THRESHOLD))
    volume_gate = float(desired.get("volumeGate", DEFAULT_VOLUME_GATE))
    set_piano_keys(int(desired.get("pianoKeys", DEFAULT_PIANO_KEYS)))

async def twin_patch_handler(patch):
    global conf_threshold, volume_gate

    if "confThreshold" in patch:
        conf_threshold = float(patch["confThreshold"])
    if "volumeGate" in patch:
        volume_gate = float(patch["volumeGate"])
    if "pianoKeys" in patch:
        set_piano_keys(int(patch["pianoKeys"]))

async def send_session_to_iothub(data):
    if not client:
        return
    message = Message(json.dumps(data))
    message.content_encoding = "utf-8"
    message.content_type = "application/json"
    message.message_id = str(uuid.uuid4())
    await client.send_message(message)


# -------- QUESTDB --------
def init_questdb():
    sql = (
        "CREATE TABLE IF NOT EXISTS piano_sessions ("
        "ts TIMESTAMP, session_id SYMBOL, total_notes INT, notes_played STRING, synced BOOLEAN"
        ") timestamp(ts);"
    )
    try:
        requests.get(config.QUESTDB_URL, params={"query": sql}, timeout=0.5)
    except Exception:
        pass


def save_to_questdb(payload, notes_played):
    synced = "true" if client is not None else "false"
    sql = (
        f"INSERT INTO piano_sessions VALUES("
        f"now(), '{payload['session_id']}', {payload['total_notes']}, '{notes_played}', {synced});"
    )
    try:
        requests.get(config.QUESTDB_URL, params={"query": sql}, timeout=0.5)
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


# ----------- MAIN ------------

async def main():
    client = await connect(config.IOT_CONNECTION_STRING)
    twin = await client.get_twin()
    load_twin_config(twin)

    client.on_twin_desired_properties_patch_received = twin_patch_handler

    while True:
        await asyncio.sleep(1)
    
    

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass