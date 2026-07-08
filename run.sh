#!/bin/bash
# Start Ollama in the background
ollama serve &
# Give it 5 seconds to wake up
sleep 5
# Run your AI routing agent!
python main.py
