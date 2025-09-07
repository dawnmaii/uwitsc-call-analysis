#!/usr/bin/env python3
"""
Ollama Analysis Script
Analyzes transcription files using Ollama and assigns scores.
"""

import os
import sys
import json
from pathlib import Path
import argparse
import time

try:
    import requests  # type: ignore
except ImportError:
    print("Warning: requests not available in current environment")
    print("This script is designed to run in the ollama_python.sif container")
    requests = None  # type: ignore

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"
SCORE_THRESHOLD = 75

def wait_for_ollama(max_wait=180):
    """Wait for Ollama server to be ready and model to be loaded"""
    print("Waiting for Ollama server to be ready...")
    server_ready = False
    model_ready = False
    
    for i in range(max_wait):
        try:
            # Check if server is responding
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                if not server_ready:
                    print("Ollama server is ready!")
                    server_ready = True
                
                # Check if model is available
                models = response.json()
                model_names = [model.get('name', '') for model in models.get('models', [])]
                if MODEL_NAME in model_names:
                    if not model_ready:
                        print(f"Model {MODEL_NAME} is available!")
                        model_ready = True
                        return True
                else:
                    if i % 15 == 0:  # Print status every 15 seconds
                        print(f"Waiting for model {MODEL_NAME} to be available... ({i}s elapsed)")
                        print(f"Available models: {model_names}")
        except Exception as e:
            if i % 10 == 0:  # Print status every 10 seconds
                print(f"Still waiting for Ollama server... ({i}s elapsed)")
        time.sleep(1)
    
    print(f"Warning: Ollama server or model not ready after {max_wait} seconds, proceeding anyway...")
    return False

def analyze_transcription_file(vtt_file, audio_file_name):
    """Analyze a single transcription file using Ollama"""
    try:
        # Read VTT file and extract text
        with open(vtt_file, 'r', encoding='utf-8') as f:
            vtt_content = f.read()
        
        # Parse VTT content to extract plain text
        lines = vtt_content.split('\n')
        transcription_lines = []
        for line in lines:
            if '-->' not in line and line.strip() and not line.startswith('WEBVTT'):
                transcription_lines.append(line.strip())
        
        transcription_text = ' '.join(transcription_lines)
        
        if not transcription_text.strip():
            print(f"No transcription text found in {vtt_file}")
            return None
        
        # Prepare the prompt
        prompt = f"""Analyze this customer service call transcription and provide a score from 0-100 based on the following criteria:

1. NetID obtained within 120 seconds (10 points)
2. Issue resolution (15 points) 
3. Quality of instructions provided (15 points)
4. Use of Zoom for verification (5 points) - Give full points if agent mentions Zoom verification at any point during the call
5. Keeping confidential information confidential until verification (7 points)
6. Overall technical support quality (48 points)

CRITICAL INSTRUCTIONS:
- Read the ENTIRE transcription word by word
- For criterion 4: Look for the exact word "Zoom" (case-insensitive) anywhere in the transcription
- If you find "Zoom" mentioned by the agent, give the full 5 points for criterion 4
- Be fair and accurate - if the agent performed well, give them the points they deserve
- Don't deduct points for minor issues or things that could have been done differently
- Focus on what the agent actually accomplished, not what they could have done better
- If the agent successfully completed all required tasks, consider giving them full points

Transcription to analyze:
{transcription_text}

Please respond in JSON format with 'score' (integer 0-100) and 'reasoning' (string explaining the score and specifically mentioning what you found about Zoom usage).

IMPORTANT: If the agent successfully completed all the main tasks (obtained NetID, resolved issue, provided instructions, used Zoom, kept info confidential), consider giving them a score of 95-100. Only deduct points for significant failures, not minor improvements that could have been made."""

        # Call Ollama API
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            },
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '')
            
            # Parse JSON response
            try:
                # Extract JSON from response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    analysis = json.loads(json_str)
                    
                    # Handle nested JSON in reasoning field
                    if 'reasoning' in analysis and isinstance(analysis['reasoning'], str):
                        try:
                            # Try to parse reasoning as JSON if it contains nested JSON
                            if analysis['reasoning'].startswith('{'):
                                nested_analysis = json.loads(analysis['reasoning'])
                                return nested_analysis.get('score', analysis.get('score', 0)), nested_analysis.get('reasoning', analysis.get('reasoning', 'No reasoning provided'))
                        except:
                            pass  # Use original analysis if nested parsing fails
                    
                    return analysis.get('score', 0), analysis.get('reasoning', 'No reasoning provided')
                else:
                    # Fallback parsing
                    score = 50  # Default score
                    reasoning = response_text
                    return score, reasoning
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                score = 50
                reasoning = response_text
                return score, reasoning
            else:
                print(f"Ollama API error: {response.status_code}")
                return 50, "Analysis failed"
            
    except Exception as e:
        print(f"Error analyzing {vtt_file}: {e}")
        return 50, "Analysis error"

def analyze_speaker_folder(speaker_folder, threshold=75):
    """Analyze all transcription files in a speaker folder"""
    speaker_path = Path(speaker_folder)
    
    if not speaker_path.exists():
        print(f"Error: Speaker folder '{speaker_folder}' does not exist")
        return {}
    
    # Find VTT files
    vtt_files = list(speaker_path.glob("*.vtt"))
    
    if not vtt_files:
        print(f"No VTT files found in {speaker_folder}")
        return {}
    
    print(f"Found {len(vtt_files)} VTT files in {speaker_folder}")
    
    # Wait for Ollama to be ready
    wait_for_ollama()
    
    # Analyze each VTT file
    results = {}
    for vtt_file in vtt_files:
        print(f"Analyzing: {vtt_file.name}")
        
        # Find corresponding audio file
        audio_file = vtt_file.with_suffix('.wav')
        if not audio_file.exists():
            # Try other extensions
            for ext in ['.mp3', '.m4a', '.flac', '.ogg']:
                audio_file = vtt_file.with_suffix(ext)
                if audio_file.exists():
                    break
        
        audio_file_name = audio_file.name if audio_file.exists() else vtt_file.stem
        
        # Analyze the transcription
        analysis_result = analyze_transcription_file(vtt_file, audio_file_name)
        
        if analysis_result:
            score, reasoning = analysis_result
            results[vtt_file.name] = {
                'audio_file': audio_file_name,
                'transcription_file': vtt_file.name,
                'score': score,
                'reasoning': reasoning,
                'transcription_preview': vtt_file.read_text(encoding='utf-8')[:200] + "..." if len(vtt_file.read_text(encoding='utf-8')) > 200 else vtt_file.read_text(encoding='utf-8')
            }
            print(f"Analysis complete - Score: {score}")
        else:
            print(f"Failed to analyze {vtt_file.name}")
        
        # Save results
        results_file = speaker_path / "analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"Saved analysis results to {results_file}")
    return results

def main():
    global MODEL_NAME
    
    if requests is None:
        print("Error: requests is not available. This script must run in the ollama_python.sif container.")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Analyze transcription files using Ollama")
    parser.add_argument("speaker_folder", help="Speaker folder containing VTT files")
    parser.add_argument("--threshold", type=int, default=SCORE_THRESHOLD, help="Score threshold for organization")
    parser.add_argument("--model", default=MODEL_NAME, help="Ollama model to use")
    
    args = parser.parse_args()
    
    # Update global model name
    MODEL_NAME = args.model
    
    # Analyze the speaker folder
    results = analyze_speaker_folder(args.speaker_folder, args.threshold)
    
    if results:
        print(f"Successfully analyzed {len(results)} transcription files")
    else:
        print("No files were analyzed successfully")

if __name__ == "__main__":
    main()
