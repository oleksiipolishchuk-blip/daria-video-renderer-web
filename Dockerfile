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

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
