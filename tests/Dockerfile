FROM python:3.11-slim
WORKDIR /app
COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt
COPY src ./src
COPY tests ./tests
ENV PYTHONPATH=/app LANGCHAIN_TRACING_V2=false
CMD ["python", "-m", "pytest", "-q"]
