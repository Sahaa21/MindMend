import asyncio
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import io
import wave
from typing import Optional
import threading
import queue


class VoiceAssistant:
    def __init__(self):
        # Faster Whisper model (small for speed, you can use 'medium' or 'large')
        print("🎤 Loading Whisper model...")
        self.model = WhisperModel("small", device="cpu", compute_type="int8")

        # Audio settings
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.CHUNK_DURATION = 0.5  # seconds
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.CHUNK_DURATION)

        # Wake word detection
        self.WAKE_WORDS = ["hello", "hey", "hi"]
        self.is_listening = False
        self.is_awake = False

        # Audio buffer
        self.audio_queue = queue.Queue()
        self.recording_buffer = []

        print("✅ Voice Assistant initialized!")

    def detect_wake_word(self, audio_data: np.ndarray) -> bool:
        """Detect wake word using Whisper"""
        try:
            # Convert numpy array to audio file in memory
            audio_bytes = self._numpy_to_wav_bytes(audio_data)

            # Transcribe
            segments, info = self.model.transcribe(
                audio_bytes,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Check for wake word
            for segment in segments:
                text = segment.text.lower().strip()
                print(f"🎯 Detected: {text}")

                for wake_word in self.WAKE_WORDS:
                    if wake_word in text:
                        return True

            return False

        except Exception as e:
            print(f"❌ Wake word detection error: {e}")
            return False

    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio to text using Faster Whisper"""
        try:
            # Convert to WAV bytes
            audio_bytes = self._numpy_to_wav_bytes(audio_data)

            # Transcribe
            segments, info = self.model.transcribe(
                audio_bytes,
                language="en",  # or None for auto-detection
                vad_filter=True
            )

            # Combine all segments
            full_text = " ".join([segment.text for segment in segments])
            return full_text.strip()

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None

    def _numpy_to_wav_bytes(self, audio_data: np.ndarray) -> io.BytesIO:
        """Convert numpy array to WAV format in memory"""
        byte_io = io.BytesIO()

        # Ensure audio is in correct format
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)

        # Write WAV file
        with wave.open(byte_io, 'wb') as wav_file:
            wav_file.setnchannels(self.CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(audio_data.tobytes())

        byte_io.seek(0)
        return byte_io

    def record_audio(self, duration: float = 5.0) -> np.ndarray:
        """Record audio from microphone"""
        print(f"🎙️ Recording for {duration} seconds...")

        recording = sd.rec(
            int(duration * self.SAMPLE_RATE),
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=np.float32
        )
        sd.wait()

        return recording.flatten()

    async def listen_for_wake_word(self):
        """Continuously listen for wake word"""
        print("👂 Listening for wake word...")

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"⚠️ Audio status: {status}")
            self.audio_queue.put(indata.copy())

        # Start audio stream
        with sd.InputStream(
                callback=audio_callback,
                channels=self.CHANNELS,
                samplerate=self.SAMPLE_RATE,
                blocksize=self.CHUNK_SIZE
        ):
            self.is_listening = True
            buffer = []

            while self.is_listening:
                try:
                    # Get audio chunk
                    audio_chunk = self.audio_queue.get(timeout=1)
                    buffer.append(audio_chunk)

                    # Keep last 2 seconds
                    if len(buffer) > int(2 / self.CHUNK_DURATION):
                        buffer.pop(0)

                    # Check for wake word every 2 seconds
                    if len(buffer) >= int(2 / self.CHUNK_DURATION):
                        audio_data = np.concatenate(buffer).flatten()

                        if self.detect_wake_word(audio_data):
                            print("🎉 Wake word detected!")
                            self.is_awake = True
                            return True

                        buffer = []

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Listen error: {e}")
                    break

        return False

    def stop_listening(self):
        """Stop listening"""
        self.is_listening = False
        self.is_awake = False


# Test function
if __name__ == "__main__":
    va = VoiceAssistant()

    print("\n🎤 Say 'Hello Sid' to activate...")
    asyncio.run(va.listen_for_wake_word())

    if va.is_awake:
        print("\n✅ Assistant activated! Recording your question...")
        audio = va.record_audio(duration=5.0)
        text = va.transcribe_audio(audio)
        print(f"\n💬 You said: {text}")