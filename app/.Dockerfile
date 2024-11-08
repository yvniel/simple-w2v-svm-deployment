FROM python:3.12.4
WORKDIR /app
COPY . /app/ 
RUN pip install -r requirements.txt
EXPOSE 8000
ENV UVICORN_CMD="uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
CMD ["sh", "-c", "$UVICORN_CMD"]
