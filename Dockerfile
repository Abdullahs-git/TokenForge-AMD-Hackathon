FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies (minimal: openai + sympy only)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY main.py router.py llm_clients.py local_solvers.py output_sanitizer.py prompt_compressor.py ./

# Create input/output mount points
RUN mkdir -p /input /output

# Unbuffered Python output for container logging
ENV PYTHONUNBUFFERED=1

# Container contract entrypoint: read /input/tasks.json -> process -> write /output/results.json -> exit 0
CMD ["python", "main.py"]
