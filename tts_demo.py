import gradio as gr
import torch
import soundfile as sf
from io import BytesIO
import os
from models import build_model, load_voice, generate_speech, list_available_voices

# Function for TTS
def synthesize_speech(text, voice):
    try:
        # Set up device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Build model and load voice
        model = build_model('kokoro-v0_19.pth', device)
        voice_data = load_voice(voice, device)
        
        # Generate speech
        audio, phonemes = generate_speech(model, text, voice_data, lang='a', device=device)
        
        if audio is not None:
            # Save audio to a .wav file in the 'outputs' folder
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "output.wav")
            sf.write(output_file, audio, samplerate=22050, format='WAV')
            return output_file, f"Generated phonemes: {phonemes}"
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
