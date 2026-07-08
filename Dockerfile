# Use a lightweight Python base image
FROM python:3.11-slim

# Install system dependencies and Ollama (added zstd!)
RUN apt-get update && apt-get install -y curl zstd
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy NER model
RUN python -m spacy download en_core_web_sm

# Start Ollama in the background, download the 1B model, and save it inside the image
RUN ollama serve & OLLAMA_PID=$! && sleep 5 && ollama pull llama3.2:1b && kill $OLLAMA_PID

# Copy all your code into the container
COPY . .

# Make the run script executable
RUN chmod +x run.sh

# When the container starts, execute the run script
CMD ["./run.sh"]
