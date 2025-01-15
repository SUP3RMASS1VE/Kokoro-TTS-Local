import torch
import os
import sys
from huggingface_hub import hf_hub_download, list_repo_files
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from importlib.util import spec_from_file_location, module_from_spec

def setup_espeak():
    """Set up espeak library paths for phonemizer."""
    try:
        # Set up espeak library paths
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        print("espeak-ng library paths set up successfully")
    except Exception as e:
        print(f"Error setting up espeak: {e}")
        raise e

def import_module_from_path(module_name, module_path):
    """Import a module from file path."""
    try:
        spec = spec_from_file_location(module_name, module_path)
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing module {module_name}: {e}")
        raise e

def build_model(model_file, device='cpu'):
    """Build the Kokoro model following official implementation."""
    try:
        # Set up espeak first
        setup_espeak()
        
        # Download necessary files from Hugging Face
        repo_id = "hexgrad/Kokoro-82M"
        model_path = hf_hub_download(repo_id=repo_id, filename="kokoro-v0_19.pth")
        kokoro_py = hf_hub_download(repo_id=repo_id, filename="kokoro.py")
        models_py = hf_hub_download(repo_id=repo_id, filename="models.py")
        istftnet_py = hf_hub_download(repo_id=repo_id, filename="istftnet.py")
        plbert_py = hf_hub_download(repo_id=repo_id, filename="plbert.py")
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        
        # Import modules in correct dependency order
        print("Importing plbert module...")
        plbert_module = import_module_from_path("plbert", plbert_py)
        print("Importing istftnet module...")
        istftnet_module = import_module_from_path("istftnet", istftnet_py)
        print("Importing models module...")
        models_module = import_module_from_path("models", models_py)
        print("Importing kokoro module...")
        kokoro_module = import_module_from_path("kokoro", kokoro_py)
        
        # Test phonemizer
        from phonemizer import phonemize
        test_phonemes = phonemize("Hello")
        print(f"Phonemizer test successful: 'Hello' -> {test_phonemes}")
        
        # Build and load the model
        print("Building model...")
        model = models_module.build_model(model_path, device)
        print(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        print(f"Error building model: {e}")
        import traceback
        traceback.print_exc()
        raise e

def load_voice(voice_name='af', device='cpu'):
    """Load a voice from the official voicepacks."""
    try:
        repo_id = "hexgrad/Kokoro-82M"
        voice_path = hf_hub_download(repo_id=repo_id, filename=f"voices/{voice_name}.pt")
        voice = torch.load(voice_path, weights_only=True).to(device)
        print(f"Loaded voice: {voice_name}")
        return voice
    except Exception as e:
        print(f"Error loading voice: {e}")
        raise e

def generate_speech(model, text, voice=None, lang='a', device='cpu'):
    """Generate speech using the Kokoro model."""
    try:
        from kokoro import generate
        audio, phonemes = generate(model, text, voice, lang=lang)
        return audio, phonemes
    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        return None, None 

def list_available_voices():
    """List all available voices from the official voicepacks."""
    try:
        repo_id = "hexgrad/Kokoro-82M"
        files = list_repo_files(repo_id)
        # Filter for voice files in the voices directory and remove .pt extension
        voices = [f.replace('voices/', '').replace('.pt', '') 
                 for f in files if f.startswith('voices/') and f.endswith('.pt')]
        return sorted(voices)
    except Exception as e:
        print(f"Error listing voices: {e}")
        return [] 