from typing import List, Dict, Optional
from datetime import datetime
import json

class ChatManager:
    """Manage chat conversation and history"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.messages: List[Dict] = []
        self.model_history: List[dict] = []  # Changed from List[Tuple[str, str]]
        self.current_audio_path: Optional[str] = None
        self.current_audio_name: Optional[str] = None
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add message to chat history
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        
        # Trim history if too long
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-(self.max_history * 2):]
    
    def update_model_history(self, history: List[dict]):
        """Update model conversation history"""
        self.model_history = history
    
    def set_audio(self, audio_path: str, audio_name: str):
        """Set current audio file context"""
        self.current_audio_path = audio_path
        self.current_audio_name = audio_name
    
    def get_messages(self) -> List[Dict]:
        """Get all chat messages"""
        return self.messages
    
    def get_model_history(self) -> List[dict]:
        """Get model conversation history"""
        return self.model_history
    
    def has_audio(self) -> bool:
        """Check if audio is loaded"""
        return self.current_audio_path is not None
    
    def clear(self):
        """Clear all chat history"""
        self.messages = []
        self.model_history = []
        self.current_audio_path = None
        self.current_audio_name = None
    
    def export_chat(self) -> str:
        """Export chat as JSON"""
        export_data = {
            "messages": self.messages,
            "audio_file": self.current_audio_name,
            "exported_at": datetime.now().isoformat()
        }
        return json.dumps(export_data, indent=2)