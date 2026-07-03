FROM python:3.12-slim

WORKDIR /app

COPY MLProject/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY MLProject/ .

EXPOSE 8080
CMD ["python", "modelling.py"]