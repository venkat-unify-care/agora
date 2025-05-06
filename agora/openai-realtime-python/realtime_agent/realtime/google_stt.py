import os
import logging
import io
try:
    from google.cloud import speech
except ImportError:
    logging.warning("Google Cloud Speech not available. Make sure to install it with: pip install google-cloud-speech")

from ..logger import setup_logger

# Set up the logger with color and timestamp support
logger = setup_logger(name=__name__, log_level=logging.INFO)

def transcribe_audio(audio_data, language_code="en-US"):
    """
    Transcribe audio using Google Speech-to-Text API.
    
    Args:
        audio_data: Raw PCM audio data (16-bit, 24kHz, mono)
        language_code: The language code to use for transcription
        
    Returns:
        str: The transcribed text, or empty string if transcription failed
    """
    try:
        # Check if Google Cloud Speech is available
        if 'google.cloud.speech' not in globals():
            logger.error("Google Cloud Speech library not available. Install with: pip install google-cloud-speech")
            return ""
            
        # Check if audio_data is not empty
        if not audio_data or len(audio_data) < 1000:  # Require at least 1000 bytes for processing
            return ""
        
        # Initialize speech client
        client = speech.SpeechClient()
        
        # Configure audio
        audio = speech.RecognitionAudio(content=audio_data)
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,  # Ensure this matches your PCM sample rate
            language_code=language_code,
            enable_automatic_punctuation=True,
            model="latest_long",  # Use enhanced model for better results
        )
        
        # Perform synchronous speech recognition
        response = client.recognize(config=config, audio=audio)
        
        # Process the response
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        
        return transcript
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return ""