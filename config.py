import os
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """Qwen2-Audio Model Configuration"""
    # Use the correct model name
    name: str = "Qwen/Qwen2-Audio-7B-Instruct"  # or "Qwen/Qwen2-Audio-7B"
    trust_remote_code: bool = True
    device_map: str = "auto"

@dataclass
class AudioConfig:
    """Audio Processing Configuration"""
    sample_rate: int = 16000
    max_duration: int = 30  # seconds
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

@dataclass
class ChatConfig:
    """Chat Configuration"""
    max_history: int = 10
    system_prompt: str = (
        "You are an expert medical AI assistant specializing in cardiac auscultation. "
        "You can listen to and analyze heart sounds. Provide detailed, professional "
        "analysis of heart rhythms, murmurs, and abnormalities. Always remind users "
        "to consult healthcare professionals for medical decisions."
    )
    welcome_message: str = (
        "👋 **Welcome to Heart Sound AI Assistant!**\n\n"
        "I can listen to and analyze heart sound recordings.\n\n"
        "**How to use:**\n"
        "1. Upload a heart sound audio file\n"
        "2. I'll listen and provide initial analysis\n"
        "3. Ask me questions about what I heard!\n\n"
        "Let's begin! Upload your heart sound recording."
    )

@dataclass
class UIConfig:
    """UI Configuration"""
    page_title: str = "Heart Sound AI Chatbot"
    page_icon: str = "🫀"
    layout: str = "wide"
    sidebar_state: str = "expanded"

@dataclass
class PathConfig:
    """Path Configuration"""
    upload_dir: str = "uploads"
    
    def __post_init__(self):
        os.makedirs(self.upload_dir, exist_ok=True)

class Config:
    """Main Configuration Class"""
    def __init__(self):
        self.model = ModelConfig()
        self.audio = AudioConfig()
        self.chat = ChatConfig()
        self.ui = UIConfig()
        self.paths = PathConfig()
    
    def get_device(self):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

# Global config instance
config = Config()