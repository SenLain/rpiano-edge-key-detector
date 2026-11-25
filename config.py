import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Directories
MODELS_DIR = BASE_DIR / "models"
RUNTIME_DIR = BASE_DIR / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# Model & Runtime IPC Paths
MODEL_PATH = MODELS_DIR / "piano_88_quant.tflite"
BASIS_PATH = MODELS_DIR / "log_freq_basis.npy"
STATE_FILE = RUNTIME_DIR / "live_state.json"
COMMAND_FILE = RUNTIME_DIR / "command.json"

# Cloud & Local Storage Credentials (loaded from .env)
IOT_CONNECTION_STRING = os.getenv("IOT_CONNECTION_STRING")
DEVICE_ID = os.getenv("DEVICE_ID")
QUESTDB_URL = os.getenv("QUESTDB_URL")

# Audio & DSP Constants
TARGET_SR = 16000
N_FFT = 8192
SAMPLES_TARGET = int(0.5 * TARGET_SR)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Default Device Twin Settings
DEFAULT_IS_LISTENING = True
DEFAULT_CONF_THRESHOLD = 0.45
DEFAULT_VOLUME_GATE = 0.0075
DEFAULT_PIANO_KEYS = 88