import gradio as gr
import torch
import soundfile as sf
from io import BytesIO
import os
from models import build_model, load_voice, generate_speech, list_available_voices
from datetime import datetime

# Function to split text into chunks
def split_text(text, max_tokens=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # Account for spaces
        if current_length + word_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Function for TTS
def synthesize_speech(text, voice):
    try:
        # Set up device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Build model and load voice
        model = build_model('kokoro-v0_19.pth', device)
        voice_data = load_voice(voice, device)
        
        # Split text into manageable chunks
        text_chunks = split_text(text, max_tokens=500)
        
        combined_audio = []
        combined_phonemes = []
        
        # Process each chunk
        for chunk in text_chunks:
            audio, phonemes = generate_speech(model, chunk, voice_data, lang='a', device=device)
            if audio is not None:
                combined_audio.extend(audio)
                combined_phonemes.append(phonemes)
        
        if combined_audio:
            # Generate a unique output filename using a timestamp
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            # Use the current timestamp for uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"output_{timestamp}.wav")
            
            # Save audio to a .wav file in the 'outputs' folder
            sf.write(output_file, combined_audio, samplerate=22050, format='WAV')
            return output_file, f"Generated phonemes: {' '.join(combined_phonemes)}"
        else:
            return None, "Failed to generate audio."

    except Exception as e:
        return None, f"Error: {e}"

# Get list of available voices
available_voices = list_available_voices()

# Gradio interface
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Kokoro TTS Demo")

        with gr.Row():
            text_input = gr.Textbox(label="Input Text", placeholder="Enter text to synthesize", lines=2)
            voice_dropdown = gr.Dropdown(label="Select Voice", choices=available_voices, value=available_voices[0])

        output_audio = gr.Audio(label="Generated Speech", type="filepath")
        output_text = gr.Textbox(label="Phonemes", interactive=False)

        def process_input(text, voice):
            audio, phonemes = synthesize_speech(text, voice)
            return audio, phonemes

        synthesize_button = gr.Button("Synthesize")
        synthesize_button.click(process_input, inputs=[text_input, voice_dropdown], outputs=[output_audio, output_text])

    demo.launch()

if __name__ == "__main__":
    main()
