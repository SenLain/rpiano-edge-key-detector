import json
import os
import streamlit as st
import config


def send_command(payload):
    tmp_path = config.COMMAND_FILE.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, config.COMMAND_FILE)


def read_live_state():
    if not config.STATE_FILE.exists():
        return None
    try:
        with open(config.STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def on_listening_change():
    send_command({"action": "set_listening", "value": st.session_state.listening_toggle})


def on_piano_size_change():
    send_command({"action": "set_piano_keys", "value": st.session_state.piano_size_select})


def render_controls(state):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Recording Song", use_container_width=True, type="primary"):
            send_command({"action": "start"})
    with col2:
        if st.button("Stop & Detect Musical Key", use_container_width=True):
            send_command({"action": "stop"})

    if "listening_toggle" not in st.session_state:
        st.session_state.listening_toggle = state["is_listening"] if state else True
    if "piano_size_select" not in st.session_state:
        initial_keys = state["config"]["pianoKeys"] if state else 88
        st.session_state.piano_size_select = initial_keys if initial_keys in config.STANDARD_PIANO_SIZES else 88

    c_listen, c_size = st.columns(2)
    with c_listen:
        st.toggle("Mic AI Listening", key="listening_toggle", on_change=on_listening_change)
    with c_size:
        st.selectbox(
            "Visual Piano Layout (Keys)",
            options=config.STANDARD_PIANO_SIZES,
            # index=config.STANDARD_PIANO_SIZES.index(st.session_state.piano_size_select) if st.session_state.piano_size_select in config.STANDARD_PIANO_SIZES else 0,
            key="piano_size_select",
            on_change=on_piano_size_change,
        )


def render_primary_status(state):
    if not state["is_listening"]:
        status_label = "PAUSED"
    elif state["recording"]:
        status_label = "RECORDING"
    else:
        status_label = "LIVE"

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", status_label)
    c2.metric("Recorded Notes", state["recorded_notes_count"])
    c3.metric("Visual Piano", f"{state['config']['pianoKeys']} keys")


def render_detected_notes(detected):
    st.subheader("Detected Notes")
    if detected:
        badges = "   ".join([f"**`{item['note']} ({item['conf']:.2f})`**" for item in detected])
        st.markdown(f"### {badges}")
    else:
        st.caption("Silent / Waiting for piano keys...")


def render_key_result(result):
    if not result:
        return
    st.divider()
    st.subheader("Last Recorded Session")
    sync_status = "Synced to IoT Hub" if result["synced"] else "Saved Locally Only (Offline)"
    st.caption(f"Session ID: `{result['session_id']}` | {sync_status} | Saved in QuestDB")
    st.write(f"**Notes Recorded ({result['total_notes']}):** `{result['notes_played']}`")

    st.markdown("**Cloud Key Classification (Mock Preview):**")
    for rank, pred in enumerate(result["mock_cloud_predictions"], start=1):
        st.write(f"#{rank} **{pred['key']}** — `{int(pred['confidence'] * 100)}%`")


def render_advanced_stats(state):
    with st.expander("Advanced Stats"):
        rms = state["rms"]
        gate = state["config"]["volumeGate"]
        st.progress(min(1.0, rms / 0.05), text=f"Mic RMS: {rms:.4f} (Gate: {gate})")

        s1, s2 = st.columns(2)
        s1.metric("TFLite Inference", f"{state['latency_ms']:.2f} ms")
        s2.metric("Azure IoT Hub", "Connected" if state["iot_connected"] else "Offline Mode")

        st.caption(f"Audio Input: `{state['mic_info']}`")
        st.json(state["config"])


@st.fragment(run_every=0.3)
def live_dashboard():
    state = read_live_state()
    if state is None:
        st.warning("Waiting for `main.py` to start...")
        return

    render_primary_status(state)
    render_detected_notes(state.get("detected", []))
    render_key_result(state.get("last_result"))
    render_advanced_stats(state)


def main():
    st.set_page_config(page_title="Edge AI Piano Monitor", layout="centered")
    st.title("Edge AI Piano Note & Key Detector")
    render_controls(read_live_state())
    st.divider()
    live_dashboard()


if __name__ == "__main__":
    main()