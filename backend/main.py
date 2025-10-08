from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import json
import os
from typing import List, Dict, Any, Optional
import base64
import io
import numpy as np
import wave


# Import voice modules
from voice_assistant import VoiceAssistant
from tts_handler import TTSHandler


app = FastAPI()


app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)


# Initialize voice components
voice_assistant = VoiceAssistant()
tts_handler = TTSHandler(engine="gtts")




class Query(BaseModel):
   question: str
   session_id: str = "default"




class VoiceQuery(BaseModel):
   audio_base64: str
   session_id: str = "default"










@app.get("/")
def read_root(request: Request):
   print("🔄 Page Reloaded from", request.client.host)
   return {"message": "Welcome to Onivah!"}




@app.post("/ask")
def ask_question(query: Query, request: Request):
   user_input = query.question.strip()
   session_id = query.session_id
   result = pq(query.question)


   # Add to history
   add_to_history(session_id, user_input, "user", result["answer"])


   # ✍️ Terminal logging
   print(f"📝 Question received from {request.client.host} [Session: {session_id}]")
   print("\n🟣 User Question:", result["original_input"])
   print("🌐 Detected Language:", result["detected_lang"])
   print("🔁 Translated to English:", result.get("translated", "[no translation]"))
   print("✅ Matched FAQ:", result["matched_question"] or "None")
   print("💬 Bot Response:", result["answer"])
   print("-" * 50)


   return {"answer": result["answer"]}




@app.post("/voice/transcribe")
async def transcribe_voice(file: UploadFile = File(...), session_id: str = "default"):
   """
   Transcribe voice audio file to text
   Supports: Tamil, English, Tulu, Kannada, Telugu, Malayalam, Hindi, Tanglish, French
   """
   try:
       print(f"🎤 Received audio file: {file.filename}, type: {file.content_type}")


       # Supported languages
       SUPPORTED_LANGUAGES = ['ta', 'en', 'kn', 'te', 'ml', 'hi', 'fr']


       # Read audio file
       audio_bytes = await file.read()


       # Save temporarily for processing
       import tempfile
       temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
       temp_file.write(audio_bytes)
       temp_file.close()


       # Transcribe directly from file path (Whisper handles multiple formats)
       # Auto-detect language from supported list
       from faster_whisper import WhisperModel
       model = voice_assistant.model
       segments, info = model.transcribe(
           temp_file.name,
           language=None,  # Auto-detect language
           vad_filter=True
       )


       # Get detected language info
       detected_language = info.language
       language_probability = info.language_probability


       # Language name mapping
       language_names = {
           'ta': 'Tamil',
           'en': 'English',
           'kn': 'Kannada',
           'te': 'Telugu',
           'ml': 'Malayalam',
           'hi': 'Hindi',
           'fr': 'French'
       }


       # Combine segments
       text = " ".join([segment.text for segment in segments]).strip()


       # Clean up temp file
       import os
       os.unlink(temp_file.name)


       if text:
           lang_name = language_names.get(detected_language, detected_language.upper())
           print(f"📝 Transcribed: {text}")
           print(f"🌐 Detected Language: {lang_name} ({detected_language}) - Confidence: {language_probability:.2%}")


           # Check if language is supported
           if detected_language not in SUPPORTED_LANGUAGES:
               print(f"⚠️ Warning: Detected language '{detected_language}' not in supported list")
               print(f"✅ Supported: Tamil, English, Kannada, Telugu, Malayalam, Hindi, French, Tanglish")


           # Process question
           result = pq(text)


           # Add detected language info to result
           print(f"🔁 NLP Detected Language: {result.get('detected_lang', 'unknown')}")
           print(f"✅ Matched FAQ: {result.get('matched_question', 'None')}")
           print(f"💬 Bot Response: {result['answer']}")
           print("-" * 50)


           # Add to history
           add_to_history(session_id, text, "user", result["answer"])


           # Generate TTS response in appropriate language
           # Use detected language for TTS (Tanglish will use 'en')
           tts_lang = detected_language if detected_language in ['ta', 'hi', 'kn', 'te', 'ml', 'fr'] else 'en'
           audio_response = tts_handler.text_to_speech_base64(result["answer"], lang=tts_lang)


           return {
               "success": True,
               "transcribed_text": text,
               "answer": result["answer"],
               "audio_response": audio_response,
               "detected_language": detected_language,
               "language_name": lang_name,
               "language_probability": language_probability
           }
       else:
           return {
               "success": False,
               "error": "Could not transcribe audio"
           }


   except Exception as e:
       print(f"❌ Voice transcription error: {e}")
       import traceback
       traceback.print_exc()
       return {
           "success": False,
           "error": str(e)
       }




@app.post("/voice/tts")
def text_to_speech(query: Query):
   """
   Convert text to speech
   Returns base64 encoded audio
   """
   try:
       text = query.question
       audio_b64 = tts_handler.text_to_speech_base64(text)


       if audio_b64:
           return {
               "success": True,
               "audio": audio_b64
           }
       else:
           return {
               "success": False,
               "error": "TTS generation failed"
           }


   except Exception as e:
       print(f"❌ TTS error: {e}")
       return {
           "success": False,
           "error": str(e)
       }


# Health check endpoint
@app.get("/health")
def health_check():
   """Health check endpoint"""
   return {
       "status": "healthy",
       "timestamp": datetime.now().isoformat(),
       "history_file_exists": os.path.exists(HISTORY_FILE),
       "total_sessions": len(chat_sessions),
       "voice_enabled": True
   }