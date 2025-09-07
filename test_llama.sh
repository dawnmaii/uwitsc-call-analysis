#!/bin/bash

# OUT_DIR is accepted as argument. Use current directory if no argument is passed.
OUT_DIR=${1:-.}

echo "Starting Ollama server..."
ollama serve &
sleep 5

echo "Pulling the model..."
ollama pull llama3.2

# Write python output and error logs in $OUT_DIR
echo "Running Python script..."
python3 llama.py > $OUT_DIR/llama.out 2> $OUT_DIR/llama.err
