from gtts import gTTS
import io
import base64
from typing import Optional
import pyttsx3


class TTSHandler:
    def __init__(self, engine="gtts"):
        """
        Initialize TTS Handler
        engine: 'gtts' (online) or 'pyttsx3' (offline)
        """
        self.engine_type = engine

        if engine == "pyttsx3":
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 0.9)  # Volume

            # Get available voices
            voices = self.engine.getProperty('voices')
            # Set to female voice if available
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break

        print(f"🔊 TTS Engine: {engine}")

    def text_to_speech_bytes(self, text: str, lang: str = "en") -> Optional[bytes]:
        """Convert text to speech and return audio bytes"""
        try:
            if self.engine_type == "gtts":
                # Google TTS (online, better quality)
                tts = gTTS(text=text, lang=lang, slow=False)

                # Save to bytes
                audio_bytes = io.BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)

                return audio_bytes.read()

            else:
                # pyttsx3 (offline)
                temp_file = "temp_audio.mp3"
                self.engine.save_to_file(text, temp_file)
                self.engine.runAndWait()

                with open(temp_file, 'rb') as f:
                    return f.read()

        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return None

    def text_to_speech_base64(self, text: str, lang: str = "en") -> Optional[str]:
        """Convert text to speech and return base64 encoded audio"""
        audio_bytes = self.text_to_speech_bytes(text, lang)

        if audio_bytes:
            return base64.b64encode(audio_bytes).decode('utf-8')
        return None

    def speak(self, text: str):
        """Directly play the audio (blocking)"""
        if self.engine_type == "pyttsx3":
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            # For gTTS, you'd need to use a player like pygame or playsound
            try:
                from playsound import playsound
                import tempfile

                audio_bytes = self.text_to_speech_bytes(text)
                if audio_bytes:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                        f.write(audio_bytes)
                        temp_path = f.name

                    playsound(temp_path)
            except ImportError:
                print("⚠️ playsound not installed. Install with: pip install playsound")


# Test
if __name__ == "__main__":
    tts = TTSHandler(engine="gtts")

    text = "Hello! How can I help you today?"

    # Get base64 audio
    audio_b64 = tts.text_to_speech_base64(text)
    print(f"✅ Audio base64 generated! Length: {len(audio_b64)}")

    # Or speak directly (uncomment to test)
    # tts.speak(text)