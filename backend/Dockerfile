FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/tmp/huggingface
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
CMD sh -c 'python -m app.init_db && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}'

