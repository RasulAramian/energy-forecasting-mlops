FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
COPY wheels/ /wheels/

RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
