# Stage 1: Base Image
# Use an official PyTorch image with CUDA support for a production-ready base.
# Using a specific version ensures reproducibility.
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# --- Metadata ---
LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="Docker image for the Cat-Dog Classifier FastAPI service."

# --- Environment Variables ---
# Prevents Python from writing .pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Ensures Python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED 1
# Set the default port for Uvicorn
ENV PORT 8000

# --- System Dependencies ---
# Update package lists and install necessary tools like git (if needed for dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends && apt-get clean && rm -rf /var/lib/apt/lists/*
# Add any system libraries your project might need here
    

# --- Application Setup ---
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching.
# If requirements.txt doesn't change, this layer won't be rebuilt.
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./app /app/app

# --- Expose Port and Define Entrypoint ---
# Expose the port the app will run on
EXPOSE 8000

# Command to run the application using Uvicorn
# The --host 0.0.0.0 makes the server accessible from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
