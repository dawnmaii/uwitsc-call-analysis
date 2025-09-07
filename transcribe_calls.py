#!/usr/bin/env python3
"""
WhisperX Transcription Script
Transcribes audio files using WhisperX and saves as VTT format.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def setup_environment():
    """Setup environment and check dependencies"""
    # Check if whisperx_script.py exists in the expected location
    script_path = "/mmfs1/gscratch/fellows/dawnmai/whisperx_script.py"
    if not os.path.exists(script_path):
        print(f"Error: whisperx_script.py not found at {script_path}")
        return False
    return True

def transcribe_speaker_folder(speaker_folder, output_format="vtt"):
    """Transcribe all audio files in a speaker folder"""
    speaker_path = Path(speaker_folder)
    
    if not speaker_path.exists():
        print(f"Error: Speaker folder '{speaker_folder}' does not exist")
        return False
    
    # Find audio files
    audio_extensions = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wmv", ".avi", ".mp4")
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(speaker_path.glob(f"*{ext}"))
    
    if not audio_files:
        print(f"No audio files found in {speaker_folder}")
        return False
    
    print(f"Found {len(audio_files)} audio files in {speaker_folder}")
    
    # Transcribe each audio file
    for audio_file in audio_files:
        print(f"Transcribing: {audio_file.name}")
        
        # Run WhisperX transcription
        cmd = [
            "python3", "/mmfs1/gscratch/fellows/dawnmai/whisperx_script.py", str(audio_file)
        ]
        
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        if result.returncode != 0:
            print(f"Transcription failed for {audio_file.name}")
            print(f"Error: {result.stderr}")
            continue
        
        print(f"Successfully transcribed {audio_file.name}")
    
    # Check for output files
    vtt_files = list(speaker_path.glob("*.vtt"))
    if vtt_files:
        print(f"Generated {len(vtt_files)} VTT files")
        return True
    else:
        print("No VTT files were generated")
        return False

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using WhisperX")
    parser.add_argument("speaker_folder", help="Speaker folder containing audio files")
    parser.add_argument("--format", default="vtt", help="Output format (default: vtt)")
    
    args = parser.parse_args()
    
    if not setup_environment():
        sys.exit(1)
    
    success = transcribe_speaker_folder(args.speaker_folder, args.format)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
