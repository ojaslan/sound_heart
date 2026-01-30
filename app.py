import streamlit as st
import os
from pathlib import Path
import time
import traceback

# Import custom modules
from config import config
from audio_processor import AudioProcessor
from model_handler import QwenAudioChatbot
from chat_manager import ChatManager

# Page configuration
st.set_page_config(
    page_title=config.ui.page_title,
    page_icon=config.ui.page_icon,
    layout=config.ui.layout,
    initial_sidebar_state=config.ui.sidebar_state
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF4B4B;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = ChatManager(max_history=config.chat.max_history)
    
    if 'audio_processor' not in st.session_state:
        st.session_state.audio_processor = AudioProcessor(
            sample_rate=config.audio.sample_rate,
            max_duration=config.audio.max_duration
        )
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = QwenAudioChatbot(
            model_name=config.model.name,
            trust_remote_code=config.model.trust_remote_code
        )
    
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if 'audio_uploaded' not in st.session_state:
        st.session_state.audio_uploaded = False
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def load_model():
    """Load the Qwen Audio model"""
    try:
        with st.spinner("🔄 Loading Qwen Audio Model... This may take a few minutes..."):
            st.session_state.chatbot.load_model()
            st.session_state.model_loaded = True
            st.success("✅ Model loaded successfully!")
            time.sleep(1)
            return True
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error(f"Details: {traceback.format_exc()}")
        return False

def process_uploaded_audio(uploaded_file):
    """Process uploaded audio file and get initial analysis"""
    try:
        st.session_state.processing = True
        
        # Save uploaded file
        file_path = os.path.join(config.paths.upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Validate audio
        is_valid, message = st.session_state.audio_processor.validate_audio_file(
            file_path, 
            config.audio.supported_formats
        )
        
        if not is_valid:
            st.error(f"❌ {message}")
            st.session_state.processing = False
            return False
        
        # Set audio context
        st.session_state.chat_manager.set_audio(file_path, uploaded_file.name)
        
        # Get initial analysis
        with st.spinner("🎧 AI is listening to your heart sound..."):
            initial_analysis = st.session_state.chatbot.initial_analysis(file_path)
        
        # Add to chat
        st.session_state.chat_manager.add_message(
            "user",
            f"[Uploaded heart sound: {uploaded_file.name}]",
            metadata={"type": "audio_upload", "file": uploaded_file.name}
        )
        
        st.session_state.chat_manager.add_message(
            "assistant",
            initial_analysis,
            metadata={"type": "initial_analysis"}
        )
        
        st.session_state.audio_uploaded = True
        st.session_state.processing = False
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
        st.error(f"Details: {traceback.format_exc()}")
        st.session_state.processing = False
        return False

def handle_user_input(user_message: str):
    """Handle user chat input"""
    if not st.session_state.chat_manager.has_audio():
        st.warning("⚠️ Please upload a heart sound recording first!")
        return
    
    st.session_state.processing = True
    
    # Add user message
    st.session_state.chat_manager.add_message("user", user_message)
    
    # Get AI response
    with st.spinner("🤔 AI is analyzing..."):
        try:
            response, updated_history = st.session_state.chatbot.chat(
                st.session_state.chat_manager.current_audio_path,
                user_message,
                st.session_state.chat_manager.get_model_history()
            )
            
            # Update histories
            st.session_state.chat_manager.update_model_history(updated_history)
            st.session_state.chat_manager.add_message("assistant", response)
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.session_state.chat_manager.add_message("assistant", error_msg)
            st.error(error_msg)
    
    st.session_state.processing = False

def display_chat():
    """Display chat messages"""
    messages = st.session_state.chat_manager.get_messages()
    
    if not messages:
        # Display welcome message
        with st.chat_message("assistant"):
            st.markdown(config.chat.welcome_message)
    else:
        # Display all messages
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

def sidebar_content():
    """Render sidebar content"""
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Model status
        st.subheader("🤖 AI Model")
        if st.session_state.model_loaded:
            st.success("✅ Model Loaded")
            st.caption(f"Model: {config.model.name}")
            st.caption(f"Device: {st.session_state.chatbot.device}")
        else:
            st.warning("⚠️ Model Not Loaded")
            if st.button("🚀 Load AI Model", type="primary", use_container_width=True):
                if load_model():
                    st.rerun()
        
        st.markdown("---")
        
        # Audio upload
        st.subheader("📤 Upload Heart Sound")
        
        uploaded_file = st.file_uploader(
            "Choose audio file",
            type=config.audio.supported_formats,
            disabled=not st.session_state.model_loaded or st.session_state.processing,
            help="Upload WAV, MP3, FLAC, OGG, or M4A format"
        )
        
        if uploaded_file and st.session_state.model_loaded:
            st.audio(uploaded_file)
            
            # Audio info
            if st.session_state.chat_manager.has_audio():
                audio_info = st.session_state.audio_processor.get_audio_info(
                    st.session_state.chat_manager.current_audio_path
                )
                
                with st.expander("📊 Audio Information"):
                    for key, value in audio_info.items():
                        st.text(f"{key}: {value}")
            
            if st.button(
                "🎧 Analyze Audio", 
                type="primary", 
                use_container_width=True,
                disabled=st.session_state.processing
            ):
                if process_uploaded_audio(uploaded_file):
                    st.success("✅ Analysis complete!")
                    time.sleep(0.5)
                    st.rerun()
        
        st.markdown("---")
        
        # Current audio status
        st.subheader("🎵 Current Audio")
        if st.session_state.chat_manager.has_audio():
            st.success("✅ Audio Loaded")
            st.caption(f"File: {st.session_state.chat_manager.current_audio_name}")
        else:
            st.info("No audio loaded")
        
        st.markdown("---")
        
        # Quick questions
        if st.session_state.audio_uploaded and not st.session_state.processing:
            st.subheader("💡 Quick Questions")
            
            quick_questions = [
                "What is the heart rate?",
                "Is the rhythm regular?",
                "Are there any murmurs?",
                "Describe the heart sounds",
                "Any abnormalities?",
                "What's your assessment?"
            ]
            
            for question in quick_questions:
                if st.button(question, use_container_width=True, key=f"quick_{question}"):
                    handle_user_input(question)
                    st.rerun()
        
        st.markdown("---")
        
        # Chat controls
        st.subheader("🔧 Chat Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_manager.clear()
                st.session_state.audio_uploaded = False
                st.rerun()
        
        with col2:
            if st.button("💾 Export", use_container_width=True):
                chat_export = st.session_state.chat_manager.export_chat()
                st.download_button(
                    label="Download JSON",
                    data=chat_export,
                    file_name=f"chat_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # Disclaimer
        st.caption("⚠️ **Medical Disclaimer**")
        st.caption(
            "This tool is for informational purposes only. "
            "Always consult qualified healthcare professionals for medical advice."
        )

def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.markdown(
        '<div class="main-header">🫀 Heart Sound AI Chatbot</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">AI-Powered Heart Sound Analysis & Consultation</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    sidebar_content()
    
    # Main chat area
    st.markdown("### 💬 Chat with AI")
    
    # Display chat
    display_chat()
    
    # Chat input
    if st.session_state.model_loaded:
        user_input = st.chat_input(
            "Ask about the heart sound..." if st.session_state.audio_uploaded
            else "Upload an audio file first...",
            disabled=not st.session_state.audio_uploaded or st.session_state.processing
        )
        
        if user_input and not st.session_state.processing:
            handle_user_input(user_input)
            st.rerun()
    else:
        st.info("👈 Please load the AI model from the sidebar to start")

if __name__ == "__main__":
    main()