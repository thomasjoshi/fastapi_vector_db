FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY requirements.txt .

# Install dependencies with options to avoid threading issues
RUN pip install --no-cache-dir --no-color --progress-bar off --disable-pip-version-check -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
