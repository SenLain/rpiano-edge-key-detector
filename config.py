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
QUESTDB_CONF = os.getenv("QUESTDB_CONF")

STANDARD_PIANO_SIZES = [25, 49, 61, 76, 88]