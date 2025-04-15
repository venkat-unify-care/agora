import time
import pyaudio
from agora_python_sdk import RtcEngine, RtcEngineEventHandler

# -------------------------
# 🔧 CONFIGURATION
# -------------------------
APP_ID = "YOUR_AGORA_APP_ID"        # 👈 Replace this
CHANNEL_NAME = "test"               # 👈 Must match what your bot uses
UID = 12345678                      # Any number
# -------------------------

class MyRtcHandler(RtcEngineEventHandler):
    def on_join_channel_success(self, channel, uid, elapsed):
        print(f"✅ Joined channel: {channel}, UID: {uid}")

    def on_error(self, err, msg):
        print(f"❌ Agora error {err}: {msg}")

    def on_warning(self, warn, msg):
        print(f"⚠️ Agora warning {warn}: {msg}")

rtc = RtcEngine()
handler = MyRtcHandler()
rtc.init(APP_ID, handler)
rtc.set_channel_profile(1)   # 1 = live broadcasting
rtc.set_client_role(1)       # 1 = broadcaster
rtc.enable_audio()

# Mic audio stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 320

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

rtc.join_channel("", CHANNEL_NAME, "", UID)
print("🎙️ Streaming mic audio... Press Ctrl+C to stop.")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        rtc.push_audio_frame(data)
        time.sleep(CHUNK / RATE)
except KeyboardInterrupt:
    print("🛑 Stopping stream...")
    stream.stop_stream()
    stream.close()
    rtc.leave_channel()
    rtc.release()
