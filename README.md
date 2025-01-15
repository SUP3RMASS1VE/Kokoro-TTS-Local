A Pinokio script of PierrunoYT's KokoroTTS 1-click installer



# Kokoro TTS Local

A local implementation of the Kokoro Text-to-Speech model, featuring dynamic module loading and automatic dependency management.

## Current Status

✅ **WORKING - READY TO USE** ✅

The project has been updated with:
- Automatic espeak-ng installation and configuration
- Dynamic module loading from Hugging Face
- Improved error handling and debugging
- Interactive CLI interface
- Cross-platform setup scripts

## Features

- Local text-to-speech synthesis using the Kokoro model
- Automatic espeak-ng setup using espeakng-loader
- Multiple voice support with easy voice selection
- Phoneme output support and visualization
- Interactive CLI for custom text input
- Voice listing functionality
- Dynamic module loading from Hugging Face
- Comprehensive error handling and logging
- Cross-platform support (Windows, Linux, macOS)

## Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- Internet connection (for initial model download)

## Dependencies

```txt
torch
phonemizer-fork
transformers
scipy
munch
soundfile
huggingface-hub
espeakng-loader
```

## Setup

### Windows
Run the PowerShell setup script:
```powershell
.\setup.ps1
```

### Linux/macOS
Run the bash setup script:
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup
If you prefer to set up manually:

1. Create a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### List Available Voices
To see all available voices from the Hugging Face repository:
```bash
python tts_demo.py --list-voices
```

### Basic Usage
Run the demo script with default text and voice:
```bash
python tts_demo.py
```

### Custom Text
Specify your own text:
```bash
python tts_demo.py --text "Your custom text here"
```

### Voice Selection
Choose a different voice (use --list-voices to see available options):
```bash
python tts_demo.py --voice "af" --text "Custom text with specific voice"
```

### Interactive Mode
If you run without any arguments, you'll be prompted to enter text interactively:
```bash
python tts_demo.py
```

The script will:
1. Download necessary model files from Hugging Face
2. Set up espeak-ng automatically using espeakng-loader
3. Import required modules dynamically
4. Test the phonemizer functionality
5. Generate speech from your text with phoneme visualization
6. Save the output as 'output.wav' (22050Hz sample rate)

## Project Structure

- `models.py`: Core model loading and speech generation functionality
  - Model building and initialization with dynamic imports
  - Voice loading and management from Hugging Face
  - Speech generation with phoneme output
  - Voice listing functionality
  - Automatic espeak-ng configuration
  - Error handling and logging
- `tts_demo.py`: Demo script showing basic usage
  - Command-line interface with argparse
  - Interactive text input mode
  - Voice selection and listing
  - Error handling and user feedback
- `setup.ps1`: Windows PowerShell setup script
  - Environment creation
  - Dependency installation
  - Automatic configuration
- `setup.sh`: Linux/macOS bash setup script
  - Environment creation
  - Dependency installation
  - Automatic configuration
- `requirements.txt`: Project dependencies

## Model Information

The project uses the Kokoro-82M model from Hugging Face:
- Repository: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- Model file: `kokoro-v0_19.pth`
- Voice files: Located in the `voices/` directory
- Supports multiple voice styles (use `--list-voices` to see available options)
- Automatically downloads required files from Hugging Face

## Technical Details

- Sample rate: 22050Hz
- Input: Text in any language (English recommended)
- Output: WAV audio file
- Dependencies are automatically managed
- Modules are dynamically loaded from Hugging Face
- Error handling includes stack traces for debugging
- Cross-platform compatibility through setup scripts

## Contributing

Feel free to contribute by:
1. Opening issues for bugs or feature requests
2. Submitting pull requests with improvements
3. Helping with documentation
4. Testing different voices and reporting issues
5. Suggesting new features or optimizations
6. Testing on different platforms and reporting results

## License

This project is licensed under the Apache 2.0 License. 
