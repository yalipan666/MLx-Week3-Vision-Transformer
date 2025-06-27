FROM python:3.10-slim

WORKDIR /appnew

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mnist_transformer.py .
COPY random_digits_on_canvas.py .
COPY app.py .
COPY db_utils.py .
COPY mnist_transformer_model.pth .

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"] 