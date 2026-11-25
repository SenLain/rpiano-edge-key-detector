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
STANDARD_PIANO_SIZES = [25, 49, 61, 76, 88]

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

    piano_keys = min(STANDARD_PIANO_SIZES, key=lambda size: abs(size - canditate))

def load_twin_config(twin):
    global conf_threshold, volume_gate, piano_keys    

    desired = get_desired_properties(twin)

    conf_threshold = float(desired.get("confThreshold", DEFAULT_CONF_THRESHOLD))
    volume_gate = float(desired.get("volumeGate", DEFAULT_VOLUME_GATE))
    set_piano_keys(int(desired.get("pianoKeys", DEFAULT_PIANO_KEYS)))

async def twin_patch_handler(patch):
    global conf_threshold, volume_gate, piano_keys

    if "confThreshold" in patch:
        conf_threshold = float(patch["confThreshold"])
    if "volumeGate" in patch:
        volume_gate = float(patch["volumeGate"])
    if "pianoKeys" in patch:
        set_piano_keys(int(patch["pianoKeys"]))

async def send_telemetry(data):
    if not client:
        return
    message = Message(json.dumps(data))
    message.content_encoding = "utf-8"
    message.content_type = "application/json"
    message.message_id = str(uuid.uuid4())
    await client.send_message(message)

async def send_session_to_iothub(record_session):
    await send_telemetry(record_session)

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