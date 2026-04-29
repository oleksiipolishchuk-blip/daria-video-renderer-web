FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Copy fonts from repo
COPY fonts/ /usr/share/fonts/truetype/montserrat/
RUN fc-cache -f -v

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY index.html .
COPY frames/ /app/frames/

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
