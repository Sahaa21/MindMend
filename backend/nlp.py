# backend/nlp.py
import json
import logging
import os
from typing import Optional

import spacy
from langdetect import detect_langs, DetectorFactory

# Install model with: python -m spacy download en_core_web_md
nlp = spacy.load("en_core_web_md")

# Deterministic language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

FAQ_DATA = []
FAQ_DOCS = []


def load_faqs():
    """Load and parse FAQ data from backend/faq.json at startup."""
    global FAQ_DATA, FAQ_DOCS
    
    faq_path = os.path.join(os.path.dirname(__file__), "faq.json")
    
    try:
        with open(faq_path, "r", encoding="utf-8") as f:
            FAQ_DATA = json.load(f)
        
        if not FAQ_DATA:
            logger.warning("FAQ file is empty")
            return
        
        # Build spaCy docs for all FAQ questions
        FAQ_DOCS = [nlp(faq["question"]) for faq in FAQ_DATA]
        logger.info(f"Loaded {len(FAQ_DATA)} FAQs")
        
    except FileNotFoundError:
        logger.error(f"FAQ file not found at {faq_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in FAQ file: {e}")
    except Exception as e:
        logger.error(f"Error loading FAQs: {e}")


def detect_language(text: str) -> tuple[str, float]:
    """
    Detect language of input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Tuple of (language_code, confidence)
    """
    try:
        results = detect_langs(text)
        if results:
            return results[0].lang, results[0].prob
        return "unknown", 0.0
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "unknown", 0.0


def match_faq(text: str, threshold: Optional[float] = None) -> dict:
    """
    Match user input to FAQ using semantic similarity.
    
    Args:
        text: User input text
        threshold: Minimum similarity threshold (uses env var if None)
        
    Returns:
        Dictionary with matched FAQ or fallback response
    """
    if threshold is None:
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
    
    # Detect language
    lang_code, lang_confidence = detect_language(text)
    
    # Check if FAQs are available
    if not FAQ_DATA or not FAQ_DOCS:
        return {
            "text": text,
            "detected_language": lang_code,
            "lang_confidence": round(lang_confidence, 4),
            "faq_id": None,
            "answer": "FAQs are not currently available. Please try again later.",
            "score": 0.0,
            "used_fallback": lang_code != "en"
        }
    
    # Process user input
    user_doc = nlp(text)
    
    # Find best matching FAQ
    best_score = 0.0
    best_idx = -1
    
    for idx, faq_doc in enumerate(FAQ_DOCS):
        similarity = user_doc.similarity(faq_doc)
        if similarity > best_score:
            best_score = similarity
            best_idx = idx
    
    # Round score
    best_score = round(best_score, 4)
    
    # Determine if we use fallback
    used_fallback = lang_code != "en"
    
    # Check if best match meets threshold
    if best_score >= threshold and best_idx >= 0:
        faq = FAQ_DATA[best_idx]
        return {
            "text": text,
            "detected_language": lang_code,
            "lang_confidence": round(lang_confidence, 4),
            "faq_id": faq["id"],
            "answer": faq["answer"],
            "score": best_score,
            "used_fallback": used_fallback
        }
    else:
        return {
            "text": text,
            "detected_language": lang_code,
            "lang_confidence": round(lang_confidence, 4),
            "faq_id": None,
            "answer": "Sorry, I don't know that. Please try rephrasing or ask a different question.",
            "score": best_score,
            "used_fallback": used_fallback
        }


# Load FAQs at module import
load_faqs()