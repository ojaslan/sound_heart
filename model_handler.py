import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from typing import List, Tuple, Optional
import logging

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
    
    def load_model(self):
        """Load Qwen2-Audio model and processor"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load processor (handles both audio and text)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            logger.info("✓ Processor loaded")
            
            # Load model - Use Qwen2AudioForConditionalGeneration, NOT AutoModelForCausalLM
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            logger.info("✓ Model loaded")
            
            # Set to evaluation mode
            self.model.eval()
            
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
        """
        Chat with the model about an audio file
        
        Args:
            audio_path: Path to audio file
            user_query: User's question
            history: Conversation history
            
        Returns:
            response: Model's response
            updated_history: Updated conversation history
        """
        if self.model is None or self.processor is None:
            raise Exception("Model not loaded. Call load_model() first.")
        
        if history is None:
            history = []
        
        try:
            # Prepare conversation with audio
            conversation = history.copy()
            
            # Add current turn with audio
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": user_query}
                ]
            })
            
            # Prepare inputs
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Process audio and text
            inputs = self.processor(
                text=text,
                audios=[audio_path],
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    max_length=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Remove input tokens from generated output
            generate_ids = generate_ids[:, inputs.input_ids.shape[1]:]
            
            # Decode response
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Update conversation history
            conversation.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response}
                ]
            })
            
            return response, conversation
            
        except Exception as e:
            logger.error(f"Error during chat: {str(e)}")
            error_msg = f"I encountered an error: {str(e)}"
            return error_msg, history
    
    def initial_analysis(self, audio_path: str) -> str:
        """
        Perform initial analysis of heart sound
        
        Args:
            audio_path: Path to heart sound audio file
            
        Returns:
            analysis: Initial analysis text
        """
        initial_prompt = (
            "Listen to this heart sound recording carefully. "
            "Please provide a detailed analysis including:\n"
            "1. Heart rate and rhythm (regular or irregular)\n"
            "2. Quality of heart sounds (S1 and S2)\n"
            "3. Presence of any murmurs, clicks, or extra sounds\n"
            "4. Any abnormalities detected\n"
            "5. Overall assessment and recommendations"
        )
        
        try:
            response, _ = self.chat(audio_path, initial_prompt, history=None)
            return response
        except Exception as e:
            return f"Error analyzing audio: {str(e)}"
    
    def analyze_specific_aspect(self, audio_path: str, aspect: str, history: List[dict]) -> Tuple[str, List[dict]]:
        """
        Analyze specific aspect of heart sound
        
        Args:
            audio_path: Path to audio file
            aspect: Specific aspect to analyze
            history: Conversation history
            
        Returns:
            response: Analysis response
            updated_history: Updated history
        """
        aspect_prompts = {
            "rhythm": "Focus on the rhythm of this heartbeat. Is it regular or irregular? Describe the pattern.",
            "rate": "What is the heart rate in this recording? Estimate the beats per minute.",
            "murmur": "Listen for any murmurs in this heart sound. If present, describe their characteristics.",
            "sounds": "Describe the heart sounds (S1 and S2) in this recording. Are they normal?",
            "abnormalities": "What abnormalities, if any, do you detect in this heart sound?"
        }
        
        prompt = aspect_prompts.get(aspect, f"Analyze the {aspect} of this heart sound.")
        
        return self.chat(audio_path, prompt, history)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.processor is not None