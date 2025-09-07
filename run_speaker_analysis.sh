#!/bin/bash
# Speaker Analysis Runner Script
# Usage: ./run_speaker_analysis.sh <base_directory> <hf_token>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_name> [threshold]"
    echo ""
    echo "Arguments:"
    echo "  folder_name       - Name of folder containing speaker folders (e.g., 'audio_data')"
    echo "  threshold         - Optional: Score threshold for organization (default: 75)"
    echo ""
    echo "Example:"
    echo "  $0 audio_data"
    echo "  $0 audio_data 80"
    echo ""
    echo "Note: HF token must be set as environment variable: export HF_TOKEN=your_token_here"
    exit 1
fi

# Parse arguments
FOLDER_NAME="$1"
THRESHOLD="${2:-75}"

# Set up paths
BASE_DIR="$(pwd)/$FOLDER_NAME"

# Use the token from environment variable
HF_TOKEN="${HF_TOKEN:-}"

echo "Starting Speaker Analysis Pipeline"
echo "Base directory: $BASE_DIR"
echo "HF Token: ${HF_TOKEN:0:10}..."
echo "Score threshold: $THRESHOLD"
echo ""

# Check if HF_TOKEN is provided
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is required"
    echo "Please set it with: export HF_TOKEN=your_token_here"
    exit 1
fi

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory '$BASE_DIR' does not exist"
    echo "   Please ensure the folder exists and contains speaker subfolders with audio files"
    exit 1
fi

echo "Using existing folder: $BASE_DIR"


# Run the orchestrator
cd /mmfs1/gscratch/fellows/dawnmai
python3 submit_slurm.py "$BASE_DIR" --hf-token "$HF_TOKEN" --threshold "$THRESHOLD"

echo ""
echo "Speaker analysis pipeline completed!"
