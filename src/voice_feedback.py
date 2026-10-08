import pyttsx3
import threading
import time

class VoiceEngine:
    def __init__(self):
        self.last_spoken_time = 0
        self.cooldown = 2.0  # Cooldown between voice alerts in seconds

    def speak_async(self, text):
        current_time = time.time()
        if current_time - self.last_spoken_time > self.cooldown:
            self.last_spoken_time = current_time
            def run():
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 160)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception:
                    pass
            threading.Thread(target=run, daemon=True).start()
