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

# Demo example text
demo_text = """In the gentle glow of the moon, a curious scene unfolded. Little creatures with wings of buttery petals fluttered silently, orchestrating a dance of shadows and light. In the heart of the garden, the flowers seemed to sway gently, participating in the quiet celebration. The little listener could almost hear a soft hum, a melody of the night, as if the stars themselves were singing, sprinkling dreams and wonder into the peaceful slumber of the world."""

# Gradio interface
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Kokoro TTS Demo")

        with gr.Row():
            text_input = gr.Textbox(label="Input Text", placeholder="Enter text to synthesize", lines=2)
            voice_dropdown = gr.Dropdown(label="Select Voice", choices=available_voices, value=available_voices[0])

        output_audio = gr.Audio(label="Generated Speech", type="filepath")
        output_text = gr.Textbox(label="Phonemes", interactive=False)

        # Function to update the input text with the demo text
        def set_demo_text():
            return demo_text

        # Create a button styled like a clickable box for demo example
        demo_box = gr.Button("Demo Text", variant="secondary")

        # When the demo box is clicked, update the input box with the demo text
        demo_box.click(set_demo_text, outputs=text_input)

        def process_input(text, voice):
            audio, phonemes = synthesize_speech(text, voice)
            return audio, phonemes

        # Change the label of the button to "Generate"
        generate_button = gr.Button("Generate")
        generate_button.click(process_input, inputs=[text_input, voice_dropdown], outputs=[output_audio, output_text])

        # Apply custom CSS to style the demo box button
        demo.css = """
        #demo_button {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            padding: 10px;
            cursor: pointer;
            text-align: center;
            display: inline-block;
            width: auto;
        }
        """

    demo.launch()

if __name__ == "__main__":
    main()
