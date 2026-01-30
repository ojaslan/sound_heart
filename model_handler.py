import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from typing import List, Tuple, Optional
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenAudioChatbot:
    """Qwen2-Audio Model Handler for Heart Sound Analysis"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code: bool = True):
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing Qwen2-Audio Chatbot on {self.device}")
    
    @st.cache_resource
    def load_model(_self):  # Note: _self instead of self for caching
        """Load Qwen2-Audio model and processor with caching"""
        try:
            logger.info(f"Loading model: {_self.model_name}")
            
            # Load processor
            _self.processor = AutoProcessor.from_pretrained(
                _self.model_name,
                trust_remote_code=_self.trust_remote_code
            )
            logger.info("✓ Processor loaded")
            
            # Load model with optimizations for CPU
            _self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                _self.model_name,
                trust_remote_code=_self.trust_remote_code,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True,  # Optimize memory usage
            )
            logger.info("✓ Model loaded")
            
            # Set to evaluation mode
            _self.model.eval()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise Exception(f"Failed to load model: {str(e)}")
    
    def chat(
        self, 
        audio_path: str, 
        user_query: str,
        history: Optional[List[dict]] = None
    ) -> Tuple[str, List[dict]]:
        """Chat with the model about an audio file"""
        if self.model is None or self.processor is None:
            raise Exception("Model not loaded. Call load_model() first.")
        
        if history is None:
            history = []
        
        try:
            # Show progress
            with st.spinner("🎧 AI is listening and analyzing..."):
                # Prepare conversation
                conversation = history.copy()
                
                # Add current turn
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": audio_path},
                        {"type": "text", "text": user_query}
                    ]
                })
                
                # Apply chat template
                text = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False
                )
                
                # Process inputs
                inputs = self.processor(
                    text=text,
                    audios=[audio_path],
                    return_tensors="pt",
                    padding=True
                )
                
                # Move to device
                inputs = inputs.to(self.device)
                
                # Generate with progress indicator
                st.info("⏳ Generating response... (This may take 1-2 minutes on CPU)")
                
                with torch.no_grad():
                    generate_ids = self.model.generate(
                        **inputs,
                        max_length=512,  # Reduced for faster generation
                        max_new_tokens=256,  # Limit output length
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        num_beams=1,  # Use greedy decoding for speed
                    )
                
                # Decode
                generate_ids = generate_ids[:, inputs.input_ids.shape[1]:]
                response = self.processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                # Update history
                conversation.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}]
                })
                
                return response, conversation
            
        except Exception as e:
            logger.error(f"Error during chat: {str(e)}")
            error_msg = f"I encountered an error: {str(e)}"
            return error_msg, history
    
    def initial_analysis(self, audio_path: str) -> str:
        """Perform initial analysis of heart sound"""
        initial_prompt = (
            "Listen to this heart sound. Provide a brief analysis covering: "
            "heart rate, rhythm (regular/irregular), S1 and S2 sounds, "
            "any murmurs or abnormalities, and overall assessment."
        )
        
        try:
            response, _ = self.chat(audio_path, initial_prompt, history=None)
            return response
        except Exception as e:
            return f"Error analyzing audio: {str(e)}"
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.processor is not None
