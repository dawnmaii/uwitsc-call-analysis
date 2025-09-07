#!/usr/bin/env python3
"""
Enhanced WhisperX Transcription Script
Performs transcription with speaker diarization and agent identification.
"""

import sys
import warnings
from pathlib import Path
import json

try:
    import whisperx  # type: ignore
except ImportError:
    print("Warning: whisperx not available in current environment")
    print("This script is designed to run in the whisperx_python.sif container")
    whisperx = None  # type: ignore

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Agent identification keywords (case-insensitive)
AGENT_KEYWORDS = [
    "service", "support", "help", "assistance", "technical", "customer",
    "agent", "representative", "specialist", "advisor", "consultant",
    "uw", "u w", "i t", "service", "center", "desk",
    "how may i assist you", "how can i help you", "this is", "good", "net id",
    "netid", "get", "start", "getting", "your", "vwnet", "v w net",
    "recovery code", "verify", "identity", "zoom", "meeting", "id number",
    "driver's license", "passport", "awesome", "all right", "take your time",
    "i see you", "i'm going to leave", "can you let me know", "thank you",
    "i'm stopping", "have a good", "rest of your day"
]

def main():
    if whisperx is None:
        print("Error: whisperx is not available. This script must run in the whisperx_python.sif container.")
        sys.exit(1)

if len(sys.argv) < 2:
        print("Usage: python whisperx_script.py <audio_file>")
    sys.exit(1)

    audio_file = sys.argv[1]
device = "cuda"
    
    # Extract agent name from folder structure
audio_path = Path(audio_file)
agent_name = audio_path.parent.name

    print(f"Processing: {audio_file}")
print(f"Agent name: {agent_name}")
    print(f"Using device: {device}")
    
    try:
    print("Loading WhisperX model...")
        model = whisperx.load_model("large-v2", device, compute_type="float16")
        print("Model loaded successfully!")

    print("Loading audio file...")
    audio = whisperx.load_audio(audio_file)
        print("Audio loaded successfully!")
    
    print("Transcribing audio...")
        result = model.transcribe(audio, batch_size=16)
        print(f"Transcription completed! Language: {result['language']}")
        print(f"Number of segments: {len(result['segments'])}")
        
        # Enhanced speaker assignment with sentence-level analysis
        print("Performing enhanced speaker diarization...")
        
        # First, try to use WhisperX's built-in diarization
        try:
            print("Loading diarization model...")
            diarize_model = whisperx.load_model("pyannote/speaker-diarization", device=device)
            print("Performing speaker diarization...")
            diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=2)
            print("Speaker detection completed!")
            
            # Assign speaker labels to words
            print("Assigning speaker labels to words...")
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Find the first speaker to say an agent keyword
            agent_speaker = None
            for segment in result["segments"]:
                speaker = segment.get('speaker', 'SPEAKER_00')
                text = segment.get('text', '').lower()
            
                # Check if this speaker said an agent keyword
                for keyword in AGENT_KEYWORDS:
                    if keyword.lower() in text:
                        if agent_speaker is None:
                            agent_speaker = speaker
                            print(f"Identified agent speaker: {speaker} (said '{keyword}')")
                    break
            
            # Update speaker labels with sentence-level analysis
            for segment in result["segments"]:
                speaker = segment.get('speaker', 'SPEAKER_00')
                text = segment.get('text', '').strip()
                
                if not text:
                    segment['speaker'] = "user"
                    continue
                
                # Split text into sentences for more granular analysis
                sentences = []
                current_sentence = ""
                for char in text:
                    current_sentence += char
                    if char in '.!?':
                        sentences.append(current_sentence.strip())
                        current_sentence = ""
                
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                
                # Analyze each sentence
                segment_speakers = []
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    is_agent = False
                    is_user = False
                    
                    # Check for agent keywords
                    for keyword in AGENT_KEYWORDS:
                        if keyword.lower() in sentence_lower:
                            is_agent = True
                            break
                    
                    # Check for user phrases
                    if not is_agent:
                        user_phrases = [
                            "my netid is", "i'm going to my laptop", 
                            "i'll open zoom", "that worked", "no, that's it", "take care"
                        ]
                        for phrase in user_phrases:
                            if phrase in sentence_lower:
                                is_user = True
                                break
                        
                        # Check for short responses
                        if not is_user and len(sentence.strip()) < 15:
                            short_responses = [
                                "yes", "no", "ok", "ok?", "yeah", "sure", "right",
                                "i can", "i will", "i have", "i do", "i am", "i'm",
                                "that's right", "exactly", "correct", "true", "false"
                            ]
                            for response in short_responses:
                                if sentence_lower.strip() == response or sentence_lower.strip().startswith(response):
                                    is_user = True
                                    break
                        
                        # Check for user repeating agent information
                        if not is_user and len(sentence.strip()) < 30:
                            repeat_patterns = [
                                
                            ]
                            for pattern in repeat_patterns:
                                if pattern in sentence_lower:
                                    is_user = True
                                    break
                        
                        # Additional user detection
                        if not is_user and len(sentence.strip()) < 10:
                            if not any(agent_word in sentence_lower for agent_word in ["service", "center", "help", "provide", "verify", "awesome", "thank"]):
                                is_user = True
                        
                        # Even more aggressive: single word responses
                        if not is_user and len(sentence.strip().split()) <= 2:
                            single_word_responses = ["yes", "no", "ok", "ok?", "yeah", "sure", "right", "good", "great", "fine"]
                            if sentence_lower.strip() in single_word_responses:
                                is_user = True
                    
                    # Assign speaker based on analysis
                    if is_agent:
                        segment_speakers.append(agent_name)
                    elif is_user:
                        segment_speakers.append("user")
                    else:
                        # Fallback to original diarization logic
                        if speaker == agent_speaker:
                            segment_speakers.append(agent_name)
            else:
                            segment_speakers.append("user")
                
                # Use the most common speaker for this segment
                from collections import Counter
                speaker_counts = Counter(segment_speakers)
                segment['speaker'] = speaker_counts.most_common(1)[0][0]
                    
        except Exception as e:
            print(f"Diarization failed: {e}")
            print("Falling back to keyword-based assignment...")
            
            # Fallback: Enhanced keyword-based assignment with sentence splitting
            for segment in result["segments"]:
                text = segment.get('text', '').strip()
                if not text:
                    segment['speaker'] = "user"
                    continue
                
                # Split text into sentences
                sentences = []
                current_sentence = ""
                for char in text:
                    current_sentence += char
                    if char in '.!?':
                        sentences.append(current_sentence.strip())
                        current_sentence = ""
                
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                
                # Assign speakers sentence by sentence
                segment_speakers = []
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    is_agent = False
                    is_user = False
                    
                    # Check for agent keywords
                    for keyword in AGENT_KEYWORDS:
                        if keyword.lower() in sentence_lower:
                            is_agent = True
                            break
                    
                    # Additional heuristics for agent identification
                    if not is_agent:
                        # Check for common agent phrases
                        agent_phrases = [
                            "service center", "how can i help", "what is your", 
                            "i can provide", "i need to verify", "are you able",
                            "if you could", "that would be great", "i see you",
                            "i'm going to leave", "can you let me know", "all right",
                            "awesome", "thank you", "i'm stopping", "take your time",
                            "recovery code", "verify your identity", "zoom application",
                            "meeting id number", "driver's license", "passport",
                            "have a good", "rest of your day", "no worries",
                            "i've just left", "did you have any other questions",
                            "vwnet id", "vw it service", "it should prompt you"
                        ]
                        for phrase in agent_phrases:
                            if phrase in sentence_lower:
                                is_agent = True
                                break
                    
                    # Check for conversational patterns - user responses after agent questions
                    if not is_agent:
                        # Look for short responses that are likely from user
                        if len(sentence.strip()) < 15:
                            short_responses = [
                                "yes", "no", "ok", "ok?", "yeah", "sure", "right",
                                "i can", "i will", "i have", "i do", "i am", "i'm",
                                "that's right", "exactly", "correct", "true", "false"
                            ]
                            for response in short_responses:
                                if sentence_lower.strip() == response or sentence_lower.strip().startswith(response):
                                    is_user = True
                                    break
                        
                        # Check for user repeating agent information (like Zoom IDs, codes, etc.)
                        if not is_user and len(sentence.strip()) < 30:
                            repeat_patterns = [
                                
                            ]
                            for pattern in repeat_patterns:
                                if pattern in sentence_lower:
                                    is_user = True
                                    break
                        
                        # Additional user detection - be more aggressive for short responses
                        if not is_user and len(sentence.strip()) < 10:
                            # If it's a very short response and not clearly agent, assume user
                            if not any(agent_word in sentence_lower for agent_word in ["service", "center", "help", "provide", "verify", "awesome", "thank"]):
                                is_user = True
                        
                        # Even more aggressive: if it's a single word response, assume user
                        if not is_user and len(sentence.strip().split()) <= 2:
                            single_word_responses = ["yes", "no", "ok", "ok?", "yeah", "sure", "right", "good", "great", "fine"]
                            if sentence_lower.strip() in single_word_responses:
                                is_user = True
                    
                    segment_speakers.append(agent_name if is_agent else "user")
                
                # Use the most common speaker for this segment
                from collections import Counter
                speaker_counts = Counter(segment_speakers)
                segment['speaker'] = speaker_counts.most_common(1)[0][0]
        
        # Save VTT file with enhanced speaker labels
        output_base = Path(audio_file).with_suffix('')
        vtt_content = "WEBVTT\n\n"
        
        # Process segments and split into more granular chunks for better speaker diarization
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"].strip()
            speaker = segment.get("speaker", "user")
            
            if not text:
                continue
            
            # Split text into sentences for more granular speaker assignment
            sentences = []
            current_sentence = ""
            for char in text:
                current_sentence += char
                if char in '.!?':
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
            
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            
            # If we have multiple sentences, try to assign speakers more granularly
            if len(sentences) > 1:
                # Calculate time per character for more accurate timing
                total_chars = len(text)
                time_per_char = (end_time - start_time) / total_chars if total_chars > 0 else 0
                
                current_time = start_time
                for i, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue
                    
                    # Determine speaker for this sentence
                    sentence_speaker = speaker  # Default to segment speaker
                    sentence_lower = sentence.lower()
                    
                    # Check for user phrases - be very specific to avoid false positives
                    user_phrases = [
                        "my netid is", "i'm going to my laptop", 
                        "i'll open zoom", "that worked", "no, that's it", "take care"
                    ]
                    is_user = False
                    for phrase in user_phrases:
                        if phrase in sentence_lower:
                            is_user = True
                            break
                    
                    # Check for conversational patterns - user responses after agent questions
                    if not is_user:
                        # Look for question patterns in previous sentences to identify user responses
                        question_indicators = [
                            "what is your", "are you able", "can you", "do you have",
                            "would you like", "is that", "did you", "have you"
                        ]
                        
                        # Check if this is a short response that might be answering a question
                        if len(sentence.strip()) < 15:
                            short_responses = [
                                "yes", "no", "ok", "ok?", "yeah", "sure", "right",
                                "i can", "i will", "i have", "i do", "i am", "i'm",
                                "that's right", "exactly", "correct", "true", "false"
                            ]
                            for response in short_responses:
                                if sentence_lower.strip() == response or sentence_lower.strip().startswith(response):
                                    is_user = True
                                    break
                        
                        # Check for user repeating agent information (like Zoom IDs, codes, etc.)
                        if not is_user and len(sentence.strip()) < 30:
                            repeat_patterns = [
                                
                            ]
                            for pattern in repeat_patterns:
                                if pattern in sentence_lower:
                                    is_user = True
                                    break
                        
                        # Additional user detection - be more aggressive for short responses
                        if not is_user and len(sentence.strip()) < 10:
                            # If it's a very short response and not clearly agent, assume user
                            if not any(agent_word in sentence_lower for agent_word in ["service", "center", "help", "provide", "verify", "awesome", "thank"]):
                                is_user = True
                        
                        # Even more aggressive: if it's a single word response, assume user
                        if not is_user and len(sentence.strip().split()) <= 2:
                            single_word_responses = ["yes", "no", "ok", "ok?", "yeah", "sure", "right", "good", "great", "fine"]
                            if sentence_lower.strip() in single_word_responses:
                                is_user = True
                    
                    # Check for agent keywords and phrases (only if not already identified as user)
                    is_agent = False
                    if not is_user:
                        for keyword in AGENT_KEYWORDS:
                            if keyword.lower() in sentence_lower:
                                is_agent = True
                                break
                        
                        # Check for agent phrases
                        if not is_agent:
                            agent_phrases = [
                                "service center", "how can i help", "what is your", 
                                "i can provide", "i need to verify", "are you able",
                                "if you could", "that would be great", "i see you",
                                "i'm going to leave", "can you let me know", "all right",
                                "awesome", "thank you", "i'm stopping", "take your time",
                                "verify your identity", "zoom application",
                                "meeting id number", "driver's license", "passport",
                                "have a good", "rest of your day", "no worries",
                                "i've just left", "did you have any other questions",
                                "vwnet id", "vw it service", "it should prompt you",
                                "recovery code", "i'll need to verify", "take a look at",
                                "driver's license", "id", "passport", "zoom and take"
                            ]
                            for phrase in agent_phrases:
                                if phrase in sentence_lower:
                                    is_agent = True
                                    break
                    
                    # Assign speaker based on analysis
                    if is_agent:
                        sentence_speaker = agent_name
                    elif is_user:
                        sentence_speaker = "user"
                    
                    # Calculate timing for this sentence
                    sentence_chars = len(sentence)
                    sentence_duration = sentence_chars * time_per_char
                    sentence_end_time = min(current_time + sentence_duration, end_time)
                    
                    # Convert to VTT format
                    start_vtt = f"{int(current_time//3600):02d}:{int((current_time%3600)//60):02d}:{current_time%60:06.3f}"
                    end_vtt = f"{int(sentence_end_time//3600):02d}:{int((sentence_end_time%3600)//60):02d}:{sentence_end_time%60:06.3f}"
                    
                    vtt_content += f"{start_vtt} --> {end_vtt}\n[{sentence_speaker}] {sentence}\n\n"
                    
                    current_time = sentence_end_time
            else:
                # Single sentence - use original timing
        start_vtt = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{start_time%60:06.3f}"
        end_vtt = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{end_time%60:06.3f}"
        
        vtt_content += f"{start_vtt} --> {end_vtt}\n[{speaker}] {text}\n\n"

    with open(f"{output_base}.vtt", 'w') as f:
        f.write(vtt_content)

    print("VTT file saved with speaker labels!")
        print("Processing completed successfully!")
        sys.exit(0)

except Exception as e:
    print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    print("Exiting with error code 1")
    sys.exit(1)

if __name__ == "__main__":
    main()
