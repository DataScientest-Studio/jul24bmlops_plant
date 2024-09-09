FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install httpx
COPY . .
EXPOSE 8004
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]