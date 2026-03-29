# Use official Python image with build tools
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for heavy libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port 10000 (Render expects this environment variable)
EXPOSE 10000

# Start the app with Gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:10000"]