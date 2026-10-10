import pyttsx3
import threading
import time
import config

class VoiceEngine:
    """
    Asynchronous non-blocking text-to-speech feedback engine using pyttsx3 with cooldown debouncing.
    """
    def __init__(self, cooldown=config.VOICE_COOLDOWN):
        self.last_spoken_time = 0
        self.cooldown = cooldown

    def speak_async(self, text):
        if not text:
            return

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
