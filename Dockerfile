# Use Python 3.11 as the base image
# TensorFlow has good compatibility with Python 3.11
FROM python:3.11-slim

# Set the working directory inside the Docker container
WORKDIR /app

# Copy requirements first
# This allows Docker to cache the installed packages
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask application and other project files
COPY . .

# Flask will listen on port 5000 inside the container
EXPOSE 5000

# Start the Flask application
CMD ["python", "app.py"]