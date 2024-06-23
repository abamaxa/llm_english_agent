# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV OPENAI_API_KEY ""

# Create a non-root user
RUN useradd -m -s /bin/bash appuser

# Set work directory
WORKDIR /app

RUN pip install --no-cache-dir poetry==1.8.3

# Copy project files
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of the application code
COPY . .

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Run the application
CMD ["python", "app/main.py"]