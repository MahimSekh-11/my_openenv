FROM python:3.11-slim

WORKDIR /app

# Install dependencies optimized for 2 vCPU / 8GB RAM limit
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy complete environment and validation code
COPY . .

# Expose Space port and start the HF Space validation server
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
