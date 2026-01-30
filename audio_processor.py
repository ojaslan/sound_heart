import librosa
import numpy as np
import soundfile as sf
import os
from typing import Tuple, Optional

class AudioProcessor:
    """Process audio files for heart sound analysis"""
    
    def __init__(self, sample_rate: int = 16000, max_duration: int = 30):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            audio: Audio data as numpy array
            sr: Sample rate
        """
        try:
            audio, sr = librosa.load(
                file_path, 
                sr=self.sample_rate,
                duration=self.max_duration
            )
            return audio, sr
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def process_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Complete audio processing pipeline
        
        Args:
            file_path: Path to audio file
            
        Returns:
            processed_audio: Processed audio data
            sample_rate: Sample rate
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        
        # Normalize
        audio = self.normalize_audio(audio)
        
        return audio, sr
    
    def get_audio_info(self, file_path: str) -> dict:
        """Get audio file information"""
        try:
            audio, sr = self.load_audio(file_path)
            duration = len(audio) / sr
            
            return {
                "duration": f"{duration:.2f} seconds",
                "sample_rate": f"{sr} Hz",
                "samples": len(audio),
                "file_size": f"{os.path.getsize(file_path) / 1024:.2f} KB"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def validate_audio_file(self, file_path: str, supported_formats: list) -> Tuple[bool, str]:
        """
        Validate audio file
        
        Returns:
            is_valid: Boolean indicating if file is valid
            message: Validation message
        """
        # Check if file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check file extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in supported_formats:
            return False, f"Unsupported format. Supported: {', '.join(supported_formats)}"
        
        # Check file size (max 50MB)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 50:
            return False, "File size too large (max 50MB)"
        
        # Try to load audio
        try:
            audio, sr = self.load_audio(file_path)
            if len(audio) == 0:
                return False, "Audio file is empty"
            return True, "Audio file is valid"
        except Exception as e:
            return False, f"Error validating audio: {str(e)}"