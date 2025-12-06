import streamlit as st
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoProcessor, MusicgenForConditionalGeneration

# SETTINGS
TOKENS_PER_SEC = 50
# We use the small model because the medium one will crash the free tier memory
MODEL_NAME = "facebook/musicgen-small"

# PAGE CONFIG
st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
    layout="centered"
)

@st.cache_resource
def load_models():
    """
    Load models into memory. Cached so it only happens once per session.
    """
    # Detect device: Streamlit Cloud Free Tier is CPU only.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 # CPU uses float32
    
    if device == "cuda":
        dtype = torch.float16

    print(f"Loading models on {device}...")

    # 1. Load Prompt Enhancer (FLAN-T5)
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    analyzer = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-base", 
        torch_dtype=dtype
    ).to(device)

    # 2. Load Music Generator (MusicGen)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    generator = MusicgenForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        torch_dtype=dtype
    ).to(device)

    return tokenizer, analyzer, processor, generator, device

def enhance_prompt(user_prompt, tokenizer, analyzer, device):
    """
    Uses T5 to expand a short description into a full music prompt.
    """
    meta_prompt = f"""
    You are a music producer. Rewrite this description into a detailed caption for a music generation AI.
    Include specific instruments, tempo, mood, and genre.
    User description: "{user_prompt}"
    Music Caption:
    """
    inputs = tokenizer(meta_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = analyzer.generate(
            **inputs, 
            max_length=128, 
            num_beams=3, 
            do_sample=True,
            temperature=0.7
        )
    
    enhanced_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return enhanced_text

def generate_audio(text_prompt, duration, processor, generator, device):
    """
    Generates the audio array.
    """
    # Estimate tokens needed (approx 50 tokens = 1 sec)
    max_tokens = int(duration * TOKENS_PER_SEC)
    
    inputs = processor(
        text=[text_prompt], 
        padding=True, 
        return_tensors="pt"
    ).to(device)

    # Generate audio
    with torch.no_grad():
        audio_values = generator.generate(
            **inputs, 
            max_new_tokens=max_tokens
        )
    
    # Extract audio and sampling rate
    sampling_rate = generator.config.audio_encoder.sampling_rate
    audio_data = audio_values.cpu().numpy().squeeze()
    
    # Normalization (prevent quiet audio)
    peak = np.max(np.abs(audio_data))
    if peak > 0:
        audio_data = audio_data / peak
    
    # Convert to 16-bit PCM for WAV format
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    
    return sampling_rate, audio_data_int16

# --- UI LAYOUT ---

st.title("🎵 AI Music Generator")
st.markdown("Enter a short idea (e.g., *'sad jazz'*), and the AI will compose a track.")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    duration = st.slider("Duration (seconds)", min_value=5, max_value=30, value=10, step=5)
    st.info("Note: Free hosting uses CPU. Generating 10s of music takes about 60-90s.")

# Main Input
user_input = st.text_area("Describe the music:", placeholder="Cyberpunk city rain, heavy synth bass, slow tempo...")

if st.button("Generate Music", type="primary"):
    if not user_input:
        st.warning("Please enter a description first.")
    else:
        # 1. Load Models
        status_text = st.empty()
        status_text.text("⏳ Loading AI models... (First run takes a minute)")
        
        try:
            tokenizer, analyzer, processor, generator, device = load_models()
            
            # 2. Enhance Prompt
            status_text.text("✨ Enhancing your prompt with AI...")
            enhanced_prompt = enhance_prompt(user_input, tokenizer, analyzer, device)
            st.success(f"**Optimized Prompt:** {enhanced_prompt}")
            
            # 3. Generate Music
            status_text.text("🎹 Composing... This will take a moment.")
            
            # Create a progress bar (fake progress because we can't track exact model steps easily)
            progress_bar = st.progress(0)
            for percent_complete in range(0, 80, 10):
                time.sleep(1) # Visual wait
                progress_bar.progress(percent_complete)

            sr, audio_data = generate_audio(enhanced_prompt, duration, processor, generator, device)
            progress_bar.progress(100)
            
            # 4. Save and Display
            filename = "generated_track.wav"
            wavfile.write(filename, sr, audio_data)
            
            status_text.text("✅ Done!")
            st.audio(filename)
            
            with open(filename, "rb") as f:
                st.download_button("Download WAV", f, file_name="ai_music.wav")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("If the app crashed, the free server might have run out of memory. Try a shorter duration.")
