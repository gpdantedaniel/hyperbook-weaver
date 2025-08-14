FROM python:3.12-slim

WORKDIR /src
COPY src /src/
COPY requirements.txt /src/
RUN apt-get update && apt-get install -y build-essential
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]