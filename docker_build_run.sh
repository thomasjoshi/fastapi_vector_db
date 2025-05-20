#!/bin/bash
# Script to build and run the FastAPI Vector DB in Docker with minimal configuration

echo "Building Docker image for FastAPI Vector DB..."
docker build -t fastapi-vector-db . || { 
  echo "Docker build failed. See error above."
  exit 1
}

echo "Running Docker container on port 8000..."
docker run -p 8000:8000 -v "$(pwd)/data:/app/data" fastapi-vector-db

# Alternative: Run with docker-compose
# echo "Running with docker-compose..."
# docker-compose up
