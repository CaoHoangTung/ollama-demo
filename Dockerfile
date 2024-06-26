FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src src

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/healthz

ENTRYPOINT ["streamlit", "run", "src/pdf_chat/chat_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]