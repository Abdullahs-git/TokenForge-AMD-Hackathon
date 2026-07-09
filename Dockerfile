FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download local deterministic NLP model for spaCy NER ($0 token cost)
RUN python -m spacy download en_core_web_sm

# Copy application source code
COPY . .

# Container contract entrypoint: read /input/tasks.json -> process -> write /output/results.json -> exit 0
CMD ["python", "main.py"]
