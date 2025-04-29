FROM python:3.12-slim

WORKDIR /conv_app

# Create a virtual environment in the container
RUN python3 -m venv .venv

# Activate the virtual environment
ENV PATH="/conv_app/.venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY app/ app/
COPY static/ ./static/
COPY .env_config .env_config

# Download the model files to make sure it's a compatible version
RUN mkdir -p /models && python ./app/deployment/download_model.py

# Create a non-root user to run the app
RUN useradd -m appuser
USER appuser

# Expose the port the app will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]